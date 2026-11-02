"""Build a DDP-WM world model from its training config and a released checkpoint (DDP-WM plan.py, load_model)."""
import hydra
import torch
from omegaconf import OmegaConf

DDPWM_TARGET = "models.ddpwm.ddp_vwm.DDPVWorldModel"


def is_ddpwm(train_cfg):
    return train_cfg.model.get("_target_") == DDPWM_TARGET


def load_ddpwm(model_ckpt, train_cfg, num_action_repeat, device):
    payload = torch.load(model_ckpt, map_location=device)
    if "dataset_info" not in payload:
        raise ValueError(f"{model_ckpt} has no 'dataset_info'; it is not a DDP-WM checkpoint")
    print(f"Loading DDP-WM state dict from epoch {payload.get('epoch', 'N/A')}: {model_ckpt}")
    motion = OmegaConf.to_container(train_cfg.motion, resolve=True)
    p_dim = train_cfg.proprio_emb_dim
    a_dim = train_cfg.action_emb_dim
    encoder = hydra.utils.instantiate(train_cfg.encoder)
    proprio_encoder = hydra.utils.instantiate(
        train_cfg.proprio_encoder, in_chans=payload["dataset_info"]["proprio_dim"], emb_dim=p_dim
    )
    action_encoder = hydra.utils.instantiate(
        train_cfg.action_encoder, in_chans=payload["dataset_info"]["action_dim"], emb_dim=a_dim
    )
    # DDP-WM passes the proprio width as d_action_embed and vice versa; both are 10, so this is harmless, and it is
    # kept as is so that the module shapes match the checkpoint exactly.
    predictor = hydra.utils.instantiate(
        train_cfg.predictor,
        num_frames=train_cfg.num_hist,
        d_visual=encoder.emb_dim,
        d_action_embed=p_dim * train_cfg.num_proprio_repeat,
        d_proprio_embed=a_dim * train_cfg.num_action_repeat,
        training_stage=train_cfg.model.training_stage,
        use_history=motion["use_history"],
        init_ckpt=None,
        num_queries=motion["num_motion_tokens"],
    )
    for name, module in (("encoder", encoder), ("proprio_encoder", proprio_encoder), ("action_encoder", action_encoder)):
        if name in payload:
            module.load_state_dict(payload[name])
    # history fusion and localizer keys are loaded by the world model (FrozenLocalizer); everything else must match
    missing, unexpected = predictor.load_state_dict(payload["predictor"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith(("history_fusion.", "localizer."))]
    if missing or unexpected:
        raise ValueError(f"DDP-WM predictor mismatch: missing {missing}, unexpected {unexpected}")
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        predictor=predictor,
        decoder=None,
        motion=motion,
        ckpt_path=str(model_ckpt),
        proprio_dim=p_dim,
        action_dim=a_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    model.eval()
    return model
