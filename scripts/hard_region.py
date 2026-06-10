from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.hard_region.cache_builder import cache_hard_regions_for_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache small GT boxes that full-image YOLO misses or scores low.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    hard_cfg = cfg.section("hard_region")

    written = cache_hard_regions_for_split(
        image_root=cfg.path_value("image_root"),
        label_root=cfg.path_value("label_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        small_area_ratio=float(hard_cfg["small_area_ratio"]),
        match_iou=float(hard_cfg["match_iou"]),
        min_detect_score=float(hard_cfg["min_detect_score"]),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"[hard] wrote {written} caches under {cfg.path_value('cache_root') / 'hard_regions' / args.split}")


if __name__ == "__main__":
    main()
