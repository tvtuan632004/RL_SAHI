from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from rl_sahi.common.actions import ACTION_NAMES
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.state_config import StateConfig, StateLayout


def save_checkpoint(
    path: Path,
    policy: QNetwork,
    state_dim: int,
    train_cfg: Any,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    layout: StateLayout | None = None,
) -> None:
    torch.save(
        {
            "model": policy.state_dict(),
            "state_dim": state_dim,
            "network_type": "spatial_cnn" if policy.use_spatial_cnn else "mlp",
            "state_layout": asdict(layout) if layout is not None else None,
            "train_cfg": asdict(train_cfg),
            "env_cfg": asdict(env_cfg),
            "state_cfg": asdict(state_cfg),
            "actions": {int(k): v for k, v in ACTION_NAMES.items()},
        },
        path,
    )


def load_policy(checkpoint_path: Path, device: torch.device | str | None = None) -> tuple[QNetwork, dict]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    env_allowed = {field.name for field in fields(EnvConfig)}
    state_allowed = {field.name for field in fields(StateConfig)}
    env_cfg = EnvConfig(**{key: value for key, value in checkpoint.get("env_cfg", {}).items() if key in env_allowed})
    state_cfg = StateConfig(**{key: value for key, value in checkpoint.get("state_cfg", {}).items() if key in state_allowed})
    hidden_dim = checkpoint.get("train_cfg", {}).get("hidden_dim", 512)
    layout_data = checkpoint.get("state_layout")
    layout = StateLayout(**layout_data) if isinstance(layout_data, dict) else None
    use_spatial_cnn = checkpoint.get("network_type") == "spatial_cnn"
    policy = QNetwork(
        int(checkpoint["state_dim"]),
        hidden_dim=hidden_dim,
        layout=layout,
        use_spatial_cnn=use_spatial_cnn,
    ).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    checkpoint["env_cfg_obj"] = env_cfg
    checkpoint["state_cfg_obj"] = state_cfg
    return policy, checkpoint
