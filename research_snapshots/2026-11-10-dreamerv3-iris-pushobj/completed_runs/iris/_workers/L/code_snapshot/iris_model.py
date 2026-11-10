import hashlib
import os
from pathlib import Path
import sys

from common import CACHE, WORKSPACE

os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
UPSTREAM = WORKSPACE / "baselines/iris-official"
sys.path.insert(0, str(UPSTREAM / "src"))

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as functional
from models.world_model import WorldModel
from models.tokenizer import lpips
from utils import init_weights


def perceptual_checkpoint(name, root, check=False):
    if name != "vgg_lpips":
        raise ValueError(name)
    path = CACHE / "iris_aux/vgg.pth"
    if hashlib.md5(path.read_bytes()).hexdigest() != "d507d7349b931f0638a25a48a722f98a":
        raise RuntimeError("IRIS LPIPS checkpoint checksum does not match upstream")
    return str(path)


lpips.get_ckpt_path = perceptual_checkpoint


def make_tokenizer(with_lpips=True):
    config = OmegaConf.load(UPSTREAM / "config/tokenizer/default.yaml")
    return hydra.utils.instantiate(config, with_lpips=with_lpips)


class ContinuousIRIS(nn.Module):
    def __init__(self):
        super().__init__()
        config = hydra.utils.instantiate(OmegaConf.load(UPSTREAM / "config/world_model/default.yaml"))
        self.world_model = WorldModel(512, 18, config)
        self.world_model.embedder.embedding_tables[0] = nn.Linear(14, config.embed_dim)
        self.world_model.embedder.embedding_tables[0].apply(init_weights)
        self.proprio_head = nn.Sequential(nn.Linear(config.embed_dim, config.embed_dim), nn.ReLU(), nn.Linear(config.embed_dim, 4))
        self.proprio_head.apply(init_weights)
        self.world_model.head_rewards.requires_grad_(False)
        self.world_model.head_ends.requires_grad_(False)

    def loss(self, tokens, actions, proprio):
        batch, length, token_count = tokens.shape
        assert token_count == 16
        observation_embeddings = self.world_model.embedder.embedding_tables[1](tokens)
        action_embeddings = self.world_model.embedder.embedding_tables[0](torch.cat([torch.tanh(actions), proprio], dim=-1))
        embeddings = torch.cat([observation_embeddings, action_embeddings[:, :, None]], dim=2).reshape(batch, length * 17, -1)
        embeddings = embeddings + self.world_model.pos_emb(torch.arange(length * 17, device=tokens.device))
        hidden = self.world_model.transformer(embeddings)
        logits = self.world_model.head_observations(hidden, num_steps=length * 17, prev_steps=0)
        labels = tokens.reshape(batch, -1)[:, 1:]
        visual_loss = functional.cross_entropy(logits[:, :-1].reshape(-1, 512).float(), labels.reshape(-1))
        proprio_prediction = self.proprio_head(hidden.reshape(batch, length, 17, -1)[:, :, 15])
        proprio_loss = functional.mse_loss(proprio_prediction.float(), proprio.float())
        return visual_loss + proprio_loss, {"visual_ce": visual_loss.detach(), "proprio_mse": proprio_loss.detach()}


class IrisScorer:
    def __init__(self, folder):
        folder = Path(folder)
        self.device = torch.device("cuda")
        self.tokenizer = make_tokenizer(with_lpips=False).to(self.device).eval()
        saved = torch.load(folder / "tokenizer_model.pt", map_location="cpu", weights_only=False)
        self.tokenizer.load_state_dict({key: value for key, value in saved["model"].items() if not key.startswith("lpips.")})
        self.model = ContinuousIRIS().to(self.device).eval()
        saved = torch.load(folder / "world_model.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(saved["model"])
        self.cache = None
        self.cache_batch = None

    @torch.no_grad()
    def prepare(self, request):
        self.first_image = torch.from_numpy(np.asarray(request["image"], dtype=np.uint8)).to(self.device).permute(2, 0, 1).float() / 255
        self.first_proprio = torch.as_tensor(request["proprio"], dtype=torch.float32, device=self.device)
        self.goal_image = torch.from_numpy(np.asarray(request["goal_image"], dtype=np.uint8)).to(self.device).permute(2, 0, 1).float() / 255
        self.goal_proprio = torch.as_tensor(request["goal_proprio"], dtype=torch.float32, device=self.device)
        self.step = int(request["step"])
        world = self.model.world_model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.tokenizer.encode(self.first_image[None], should_preprocess=True).tokens
            prefix_cache = world.transformer.generate_empty_keys_values(1, 101)
            embeddings = world.embedder.embedding_tables[1](tokens) + world.pos_emb(torch.arange(16, device=self.device))
            world.transformer(embeddings, prefix_cache)
        self.prefix = [(cache.get()[0].clone(), cache.get()[1].clone()) for cache in prefix_cache._keys_values]

    @torch.no_grad()
    def score(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        count, horizon = actions.shape[:2]
        assert horizon == 5
        world = self.model.world_model
        if self.cache is None or self.cache_batch != count:
            self.cache = world.transformer.generate_empty_keys_values(count, 101)
            self.cache_batch = count
        for cache, (keys, values) in zip(self.cache._keys_values, self.prefix):
            cache._k_cache._cache[:, :, :16].copy_(keys.expand(count, -1, -1, -1))
            cache._v_cache._cache[:, :, :16].copy_(values.expand(count, -1, -1, -1))
            cache._k_cache._size = 16
            cache._v_cache._size = 16
        proprio = self.first_proprio[None].expand(count, -1)
        costs = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for model_step in range(horizon):
                position = self.cache.size
                embedding = world.embedder.embedding_tables[0](torch.cat([torch.tanh(actions[:, model_step]), proprio.float()], dim=-1))[:, None]
                hidden = world.transformer(embedding + world.pos_emb(torch.tensor([position], device=self.device)), self.cache)
                logits = world.head_observations(hidden, num_steps=1, prev_steps=position)
                tokens = []
                for token_index in range(16):
                    token = logits[:, -1].argmax(-1)
                    tokens.append(token)
                    position = self.cache.size
                    embedding = world.embedder.embedding_tables[1](token)[:, None]
                    hidden = world.transformer(embedding + world.pos_emb(torch.tensor([position], device=self.device)), self.cache)
                    if token_index < 15:
                        logits = world.head_observations(hidden, num_steps=1, prev_steps=position)
                tokens = torch.stack(tokens, dim=1)
                quantized = self.tokenizer.embedding(tokens).reshape(count, 4, 4, 512).permute(0, 3, 1, 2).contiguous()
                images = self.tokenizer.decode(quantized).float()
                proprio = self.model.proprio_head(hidden[:, -1]).float()
                visual_cost = (images - (self.goal_image[None] * 2 - 1)).square().mean(dim=(1, 2, 3))
                proprio_cost = (proprio - self.goal_proprio[None]).square().mean(dim=1)
                costs.append(visual_cost + proprio_cost)
        future = torch.stack(costs, dim=1)
        if self.step < horizon:
            return future[:, -1].cpu().numpy()
        initial = (2 * (self.first_image - self.goal_image)).square().mean() + (self.first_proprio - self.goal_proprio).square().mean()
        costs = torch.cat([initial.expand(count, 1), future], dim=1)
        weights = 2.0 ** torch.arange(horizon + 1, device=self.device)
        return (costs * (weights / weights.sum())[None]).mean(dim=1).cpu().numpy()
