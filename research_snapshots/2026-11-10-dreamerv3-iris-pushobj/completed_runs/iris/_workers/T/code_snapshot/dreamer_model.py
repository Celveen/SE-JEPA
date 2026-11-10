import sys
from pathlib import Path

import numpy as np
import yaml

from common import WORKSPACE


UPSTREAM = WORKSPACE / "baselines/dreamerv3-official"
sys.path.insert(0, str(UPSTREAM))

import elements
import embodied.jax
import embodied.jax.nets as networks
import jax
import jax.numpy as jnp
import ninjax as nj
from dreamerv3 import rssm
from dreamerv3.agent import Agent


def official_config():
    config = yaml.safe_load((UPSTREAM / "dreamerv3/configs.yaml").read_text())["defaults"]["agent"]
    for component in ["enc", "dec"]:
        config[component]["simple"]["mults"] = tuple(config[component]["simple"]["mults"])
    for field in ["lr", "agc", "eps", "beta1", "beta2", "wd"]:
        config["opt"][field] = float(config["opt"][field])
    config["offline_action_encoding"] = "tanh(physical relative action)"
    return config


class OfflineDreamer(nj.Module):
    def __init__(self, config):
        self.config = config
        spaces = {"image": elements.Space(np.uint8, (64, 64, 3)), "proprio": elements.Space(np.float32, (4,))}
        actions = {"action": elements.Space(np.float32, (10,), -1, 1)}
        self.enc = rssm.Encoder(spaces, **config["enc"]["simple"], name="enc")
        self.dyn = rssm.RSSM(actions, **config["dyn"]["rssm"], name="dyn")
        self.dec = rssm.Decoder(spaces, **config["dec"]["simple"], name="dec")
        optimizer = Agent._make_opt(None, **config["opt"])
        self.opt = embodied.jax.Optimizer([self.enc, self.dyn, self.dec], optimizer, name="opt")

    def loss(self, batch, training=True):
        batch_size, length = batch["proprio"].shape[:2]
        reset = jnp.zeros((batch_size, length), dtype=bool).at[:, 0].set(True)
        encoded_actions = jnp.tanh(batch["action"])
        previous = jnp.concatenate([jnp.zeros_like(encoded_actions[:, :1]), encoded_actions[:, :-1]], axis=1)
        observations = {"image": batch["image"], "proprio": batch["proprio"]}
        _, _, tokens = self.enc(self.enc.initial(batch_size), observations, reset, training)
        _, _, losses, features, metrics = self.dyn.loss(self.dyn.initial(batch_size), tokens, {"action": previous}, reset, training)
        _, _, reconstructions = self.dec(self.dec.initial(batch_size), features, reset, training)
        losses["image"] = reconstructions["image"].loss(jnp.float32(batch["image"]) / 255)
        losses["proprio"] = reconstructions["proprio"].loss(batch["proprio"])
        scales = self.config["loss_scales"]
        total = sum(value.mean() * (scales["rec"] if name in {"image", "proprio"} else scales[name]) for name, value in losses.items())
        metrics.update({"loss/" + name: value.mean() for name, value in losses.items()})
        return jnp.float32(total), metrics

    def train(self, batch):
        optimizer_metrics, metrics = self.opt(self.loss, batch, training=True, has_aux=True)
        return {**optimizer_metrics, **metrics}

    def prepare(self, image, proprio):
        image = image[None]
        proprio = proprio[None]
        reset = jnp.ones((1,), dtype=bool)
        _, _, tokens = self.enc(self.enc.initial(1), {"image": image, "proprio": proprio}, reset, False, single=True)
        carry, _, _ = self.dyn.observe(self.dyn.initial(1), tokens, {"action": jnp.zeros((1, 10), jnp.float32)}, reset, False, single=True)
        return carry

    def score(self, carry, actions, goal_image, goal_proprio, first_image, first_proprio, step):
        count, horizon = actions.shape[:2]
        repeated = jax.tree.map(lambda value: jnp.repeat(value, count, axis=0), carry)
        _, features, _ = self.dyn.imagine(repeated, {"action": jnp.tanh(actions)}, horizon, False)
        reset = jnp.zeros((count, horizon), dtype=bool)
        _, _, observations = self.dec(self.dec.initial(count), features, reset, False)
        images = jnp.float32(observations["image"].pred())
        proprio = jnp.float32(observations["proprio"].pred())
        image_cost = jnp.mean(jnp.square(2 * (images - jnp.float32(goal_image)[None, None] / 255)), axis=(2, 3, 4))
        proprio_cost = jnp.mean(jnp.square(proprio - goal_proprio[None, None]), axis=-1)
        future = image_cost + proprio_cost
        initial = jnp.mean(jnp.square(2 * (jnp.float32(first_image) - jnp.float32(goal_image)) / 255)) + jnp.mean(jnp.square(first_proprio - goal_proprio))
        costs = jnp.concatenate([jnp.full((count, 1), initial), future], axis=1)
        weights = jnp.float32(2) ** jnp.arange(horizon + 1)
        weighted = jnp.mean(costs * (weights / weights.sum())[None], axis=1)
        return jnp.where(step < horizon, future[:, -1], weighted)


class DreamerScorer:
    def __init__(self, checkpoint):
        import pickle
        with Path(checkpoint).open("rb") as handle:
            saved = pickle.load(handle)
        self.model = OfflineDreamer(saved["config"], name="wm")
        parameters = {key: value for key, value in saved["state"].items() if not key.startswith("wm/opt/")}
        self.parameters = jax.device_put(parameters)
        self.prepare_function = jax.jit(nj.pure(self.model.prepare))
        self.score_function = jax.jit(nj.pure(self.model.score))

    def prepare(self, request):
        self.first_image = np.asarray(request["image"], dtype=np.uint8)
        self.first_proprio = np.asarray(request["proprio"], dtype=np.float32)
        self.goal_image = np.asarray(request["goal_image"], dtype=np.uint8)
        self.goal_proprio = np.asarray(request["goal_proprio"], dtype=np.float32)
        self.step = np.int32(request["step"])
        _, self.carry = self.prepare_function(self.parameters, self.first_image, self.first_proprio, seed=0)

    def score(self, actions):
        _, losses = self.score_function(self.parameters, self.carry, np.asarray(actions, np.float32), self.goal_image, self.goal_proprio, self.first_image, self.first_proprio, self.step, seed=1)
        return np.asarray(losses)
