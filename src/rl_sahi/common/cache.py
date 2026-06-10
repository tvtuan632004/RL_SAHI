from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import image_id


DETECTION_CACHE_VERSION = 2


@dataclass(slots=True)
class DetectionCache:
    image_path: str
    image_shape: tuple[int, int]
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    feature: np.ndarray
    feature_layers: tuple[int, ...]
    objectness_map: np.ndarray
    spatial_feature_map: np.ndarray


@dataclass(slots=True)
class HardRegionCache:
    image_path: str
    image_shape: tuple[int, int]
    hard_boxes: np.ndarray
    small_gt_boxes: np.ndarray
    gt_boxes: np.ndarray
    matched_iou: np.ndarray
    matched_score: np.ndarray


def detection_cache_path(cache_root: Path, split: str, image_path: Path) -> Path:
    return Path(cache_root) / "detections" / split / f"{image_id(image_path)}.npz"


def hard_region_cache_path(cache_root: Path, split: str, image_path: Path) -> Path:
    return Path(cache_root) / "hard_regions" / split / f"{image_id(image_path)}.npz"


def save_detection_cache(path: Path, cache: DetectionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(DETECTION_CACHE_VERSION, dtype=np.int32),
        image_path=np.asarray(cache.image_path),
        image_shape=np.asarray(cache.image_shape, dtype=np.int32),
        boxes=cache.boxes.astype(np.float32),
        scores=cache.scores.astype(np.float32),
        classes=cache.classes.astype(np.float32),
        feature=cache.feature.astype(np.float32),
        feature_layers=np.asarray(cache.feature_layers, dtype=np.int32),
        objectness_map=cache.objectness_map.astype(np.float32),
        spatial_feature_map=cache.spatial_feature_map.astype(np.float32),
    )


def detection_cache_is_current(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        if "cache_version" not in data.files:
            return False
        version = int(np.asarray(data["cache_version"]).item())
        return (
            version >= DETECTION_CACHE_VERSION
            and "objectness_map" in data.files
            and "spatial_feature_map" in data.files
        )


def load_detection_cache(path: Path) -> DetectionCache:
    with np.load(path, allow_pickle=False) as data:
        shape = data["image_shape"].astype(np.int32).tolist()
        objectness_map = (
            data["objectness_map"].astype(np.float32)
            if "objectness_map" in data.files
            else np.zeros((0,), dtype=np.float32)
        )
        spatial_feature_map = (
            data["spatial_feature_map"].astype(np.float32)
            if "spatial_feature_map" in data.files
            else np.zeros((0,), dtype=np.float32)
        )
        return DetectionCache(
            image_path=str(data["image_path"].item()),
            image_shape=(int(shape[0]), int(shape[1])),
            boxes=data["boxes"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            classes=data["classes"].astype(np.float32),
            feature=data["feature"].astype(np.float32),
            feature_layers=tuple(int(x) for x in data["feature_layers"].tolist()),
            objectness_map=objectness_map,
            spatial_feature_map=spatial_feature_map,
        )


def save_hard_region_cache(path: Path, cache: HardRegionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_path=np.asarray(cache.image_path),
        image_shape=np.asarray(cache.image_shape, dtype=np.int32),
        hard_boxes=cache.hard_boxes.astype(np.float32),
        small_gt_boxes=cache.small_gt_boxes.astype(np.float32),
        gt_boxes=cache.gt_boxes.astype(np.float32),
        matched_iou=cache.matched_iou.astype(np.float32),
        matched_score=cache.matched_score.astype(np.float32),
    )


def load_hard_region_cache(path: Path) -> HardRegionCache:
    data = np.load(path, allow_pickle=False)
    shape = data["image_shape"].astype(np.int32).tolist()
    return HardRegionCache(
        image_path=str(data["image_path"].item()),
        image_shape=(int(shape[0]), int(shape[1])),
        hard_boxes=data["hard_boxes"].astype(np.float32),
        small_gt_boxes=data["small_gt_boxes"].astype(np.float32),
        gt_boxes=data["gt_boxes"].astype(np.float32),
        matched_iou=data["matched_iou"].astype(np.float32),
        matched_score=data["matched_score"].astype(np.float32),
    )
