"""LeWorldModel (LeWM; Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026), ported as a baseline.

LeWM is trained end to end from pixels with two losses: next-embedding MSE and the SIGReg regularizer. Here it is
trained on our datasets (train_lewm.py) and planned with our MPC stack, so it is evaluated exactly like our models.

* SIGReg, FeedForward, Attention, ConditionalBlock, Block, Transformer, Embedder, MLP and ARPredictor are copied
  from https://github.com/lucas-maes/le-wm (module.py, commit 8edfeb3, MIT license); `LeWMWorldModel.embed`,
  `predict` and `rollout` follow its jepa.py.
* LeWM's encoder is HuggingFace's ViTModel built by stable_pretraining.backbone.utils.vit_hf("tiny", patch_size=14,
  image_size=224, pretrained=False), i.e. ViTConfig(hidden_size=192, num_hidden_layers=12, num_attention_heads=3,
  intermediate_size=768) without pooling layer, read at the [CLS] token. transformers is not installed in our env,
  so `ViTTiny` reimplements that module (pre-LN blocks, exact GELU, qkv bias, LayerNorm eps 1e-12, HF's
  truncated-normal init with std 0.02).
* LeWM has no proprioceptive input. Our planners' objective also compares proprio embeddings, so `encode_obs` and
  `rollout` return a constant zero proprio channel, which contributes nothing to the cost.
"""
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

LEWM_TARGET = "models.lewm.LeWMWorldModel"


# ----------------------------------------------------------------------------------------------- le-wm module.py

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


# ------------------------------------------------------------------------------------- encoder (HF ViTModel tiny)

class _ViTLayer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, eps):
        super().__init__()
        self.heads = heads
        self.layernorm_before = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.layernorm_after = nn.LayerNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in self.qkv(self.layernorm_before(x)).chunk(3, -1))
        x = x + self.proj(rearrange(F.scaled_dot_product_attention(q, k, v), "b h n d -> b n (h d)"))
        return x + self.fc2(F.gelu(self.fc1(self.layernorm_after(x))))


class ViTTiny(nn.Module):
    """HF ViTModel (no pooler) at ViT-Tiny size; forward returns the final-LayerNorm [CLS] token."""

    def __init__(self, image_size=224, patch_size=14, dim=192, depth=12, heads=3, mlp_dim=768, eps=1e-12):
        super().__init__()
        self.image_size, self.patch_size = image_size, patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, dim))
        self.layers = nn.ModuleList([_ViTLayer(dim, heads, mlp_dim, eps) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=eps)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        assert x.shape[-1] == self.image_size and x.shape[-2] == self.image_size, x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)[:, 0]


# ------------------------------------------------------------------------------------------------ world model

class LeWMWorldModel(nn.Module):
    """LeWM with the world-model interface our planners use (encode_obs / rollout / decoder)."""

    def __init__(self, action_dim, history_size=3, img_size=224, patch_size=14, embed_dim=192,
                 pred_depth=6, pred_heads=16, pred_mlp_dim=2048, pred_dim_head=64, pred_dropout=0.1,
                 proj_hidden_dim=2048):
        super().__init__()
        self.history_size = history_size
        self.num_hist = history_size
        self.decoder = None
        self.encoder = ViTTiny(image_size=img_size, patch_size=patch_size, dim=embed_dim)
        self.predictor = ARPredictor(
            num_frames=history_size, input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim,
            depth=pred_depth, heads=pred_heads, mlp_dim=pred_mlp_dim, dim_head=pred_dim_head,
            dropout=pred_dropout, emb_dropout=0.0,
        )
        self.action_encoder = Embedder(input_dim=action_dim, emb_dim=embed_dim)
        self.projector = MLP(embed_dim, proj_hidden_dim, embed_dim, norm_fn=nn.BatchNorm1d)
        self.pred_proj = MLP(embed_dim, proj_hidden_dim, embed_dim, norm_fn=nn.BatchNorm1d)

    def embed(self, visual):
        """visual: (b, t, 3, H, W) -> (b, t, D)."""
        b = visual.shape[0]
        cls = self.encoder(rearrange(visual, "b t ... -> (b t) ..."))
        return rearrange(self.projector(cls), "(b t) d -> b t d", b=b)

    def predict(self, emb, act_emb):
        """emb, act_emb: (b, t, D) -> next-step predictions (b, t, D)."""
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def loss(self, visual, act, sigreg, lambd, num_preds=1):
        """LeWM training loss (le-wm train.py, lejepa_forward) on a window of history_size + num_preds frames."""
        emb = self.embed(visual)
        act_emb = self.action_encoder(torch.nan_to_num(act, 0.0))
        pred = self.predict(emb[:, : self.history_size], act_emb[:, : self.history_size])
        pred_loss = (pred - emb[:, num_preds:]).pow(2).mean()
        sigreg_loss = sigreg(emb.transpose(0, 1))
        return pred_loss + lambd * sigreg_loss, pred_loss, sigreg_loss

    @staticmethod
    def _as_obs(emb):
        return {"visual": emb.unsqueeze(2), "proprio": emb.new_zeros(*emb.shape[:2], 1)}

    def encode_obs(self, obs):
        return self._as_obs(self.embed(obs["visual"]))

    def rollout(self, obs_0, act):
        """obs_0["visual"]: (b, n, 3, H, W); act: (b, n + T, action_dim) -> z_obses with n + T + 1 steps, emb.

        Same recursion as le-wm JEPA.rollout: each step predicts from the last history_size embeddings and their
        actions. The action embedder works per step, so embedding the whole sequence once is equivalent.
        """
        n = obs_0["visual"].shape[1]
        emb = self.embed(obs_0["visual"])
        act_emb = self.action_encoder(act)
        hs = self.history_size
        for t in range(act.shape[1] - n + 1):
            cur = n + t
            pred = self.predict(emb[:, -hs:], act_emb[:, max(0, cur - hs):cur])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
        return self._as_obs(emb), emb


def is_lewm(train_cfg):
    return train_cfg.model.get("_target_") == LEWM_TARGET


def build_lewm(model_cfg):
    kw = {k: v for k, v in dict(model_cfg).items() if k != "_target_"}
    return LeWMWorldModel(**kw)


def load_lewm(model_ckpt, train_cfg, device):
    payload = torch.load(model_ckpt, map_location="cpu")
    model = build_lewm(train_cfg.model)
    model.load_state_dict(payload["model"])
    print(f"Loading LeWM from epoch {payload.get('epoch', 'N/A')}: {model_ckpt}")
    return model.to(device).eval()
