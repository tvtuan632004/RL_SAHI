from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.trainer import TrainConfig, train_dqn


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN to choose one adaptive slice from cached YOLO state.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    train_cfg = cfg.dataclass_instance("train", TrainConfig)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    if args.episodes is not None:
        train_cfg.episodes = args.episodes

    checkpoint = train_dqn(
        image_root=cfg.path_value("image_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        out_dir=cfg.path_value("dqn_out_dir"),
        cfg=train_cfg,
        env_cfg=env_cfg,
        state_cfg=state_cfg,
        limit=args.limit,
        device_name=args.device or cfg.optional_str("train", "device"),
    )
    print(f"[train] best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
