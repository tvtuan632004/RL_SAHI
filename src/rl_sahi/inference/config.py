from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InferenceConfig:
    full_imgsz: int = 640
    slice_imgsz: int = 640
    full_conf: float = 0.01
    output_conf: float = 0.3
    iou: float = 0.7
    merge_iou: float = 0.5
    max_det: int = 3000
    max_slices: int = 0
    device: str | None = None
    feature_layers: tuple[int, ...] = (10,)
    min_slice_detections: int = 1
    max_slice_attempts: int = 0
