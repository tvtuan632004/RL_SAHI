from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.data import read_image


DEFAULT_CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], thickness: int = 1) -> None:
    for box in as_boxes(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def class_label(class_id: float, score: float, class_names=DEFAULT_CLASS_NAMES) -> str:
    idx = int(class_id)
    if isinstance(class_names, dict):
        name = str(class_names.get(idx, f"class_{idx}"))
    else:
        name = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
    return f"{name} {float(score):.2f}"


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y1 = max(0, y - th - baseline - 4)
    y2 = y1 + th + baseline + 4
    x2 = min(image.shape[1] - 1, x + tw + 6)
    cv2.rectangle(image, (x, y1), (x2, y2), color, -1)
    cv2.putText(image, text, (x + 3, y2 - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
    class_names=DEFAULT_CLASS_NAMES,
    full_color: tuple[int, int, int] = (0, 190, 0),
    slice_color: tuple[int, int, int] = (255, 120, 0),
) -> None:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return
    for box, score, cls, source in zip(boxes, scores, classes, sources):
        color = full_color if int(source) == 0 else slice_color
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        draw_label(image, class_label(cls, score, class_names), x1, y1, color)


def save_inference_visual(
    image_path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
    accepted_rois: np.ndarray,
    rejected_rois: np.ndarray,
    out_path: Path,
    class_names=DEFAULT_CLASS_NAMES,
) -> None:
    image = read_image(image_path)
    draw_detections(image, boxes, scores, classes, sources, class_names=class_names)
    draw_boxes(image, rejected_rois, (0, 165, 255), thickness=2)
    draw_boxes(image, accepted_rois, (0, 0, 255), thickness=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
