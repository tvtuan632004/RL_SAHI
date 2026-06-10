from __future__ import annotations

from pathlib import Path

from rl_sahi.common.cache import (
    detection_cache_path,
    hard_region_cache_path,
    load_detection_cache,
    save_hard_region_cache,
)
from rl_sahi.common.data import iter_images
from rl_sahi.hard_region.regions import build_hard_region_cache


def cache_hard_regions_for_split(
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    split: str,
    small_area_ratio: float = 0.01,
    match_iou: float = 0.4,
    min_detect_score: float = 0.5,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    images = iter_images(image_root, split=split, limit=limit)
    written = 0
    for index, image_path in enumerate(images, start=1):
        det_path = detection_cache_path(cache_root, split, image_path)
        if not det_path.exists():
            raise FileNotFoundError(f"Missing detection cache: {det_path}. Run scripts/detect.py first.")
        out_path = hard_region_cache_path(cache_root, split, image_path)
        if out_path.exists() and not overwrite:
            continue
        det = load_detection_cache(det_path)
        hard = build_hard_region_cache(
            image_path=image_path,
            image_root=image_root,
            label_root=label_root,
            detection_boxes=det.boxes,
            detection_scores=det.scores,
            image_shape=det.image_shape,
            small_area_ratio=small_area_ratio,
            match_iou=match_iou,
            min_detect_score=min_detect_score,
        )
        save_hard_region_cache(out_path, hard)
        written += 1
        if index == 1 or index % 50 == 0:
            print(f"[hard] {split}: {index}/{len(images)} cached -> {out_path}")
    return written
