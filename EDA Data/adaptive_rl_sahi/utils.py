from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

VISDRONE_CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

SIZE_SMALL = 32 * 32
SIZE_MEDIUM = 96 * 96


@dataclass(frozen=True)
class BoxRecord:
    image_id: str
    split: str
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    image_width: int
    image_height: int

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def norm_center_x(self) -> float:
        return self.center_x / self.image_width

    @property
    def norm_center_y(self) -> float:
        return self.center_y / self.image_height

    @property
    def norm_width(self) -> float:
        return self.width / self.image_width

    @property
    def norm_height(self) -> float:
        return self.height / self.image_height

    @property
    def size_bucket(self) -> str:
        if self.area < SIZE_SMALL:
            return "small"
        if self.area < SIZE_MEDIUM:
            return "medium"
        return "large"

    def to_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


@dataclass(frozen=True)
class PredictionRecord:
    image_id: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int
    image_height: int

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def size_bucket(self) -> str:
        if self.area < SIZE_SMALL:
            return "small"
        if self.area < SIZE_MEDIUM:
            return "medium"
        return "large"

    def to_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_paths(split: str) -> tuple[Path, Path, Path, Path]:
    raw_split = RAW_DATA_ROOT / f"VisDrone2019-DET-{split}"
    processed_images = PROCESSED_DATA_ROOT / "images" / split
    processed_labels = PROCESSED_DATA_ROOT / "labels" / split
    return raw_split / "images", raw_split / "annotations", processed_images, processed_labels


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def read_yolo_labels(split: str) -> list[BoxRecord]:
    _, _, image_dir, label_dir = split_paths(split)
    records: list[BoxRecord] = []
    if not label_dir.exists():
        return records

    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = image_dir / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        width, height = image_size(image_path)
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
            abs_w = bw * width
            abs_h = bh * height
            x1 = (cx * width) - abs_w / 2
            y1 = (cy * height) - abs_h / 2
            records.append(
                BoxRecord(
                    image_id=label_path.stem,
                    split=split,
                    class_id=class_id,
                    class_name=VISDRONE_CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    x1=x1,
                    y1=y1,
                    x2=x1 + abs_w,
                    y2=y1 + abs_h,
                    width=abs_w,
                    height=abs_h,
                    image_width=width,
                    image_height=height,
                )
            )
    return records


def read_all_yolo_labels(splits: Iterable[str]) -> list[BoxRecord]:
    records: list[BoxRecord] = []
    for split in splits:
        records.extend(read_yolo_labels(split))
    return records


def validate_yolo_dataset(splits: Iterable[str]) -> dict[str, object]:
    result: dict[str, object] = {"splits": {}, "invalid": []}
    invalid: list[dict[str, object]] = []
    for split in splits:
        _, _, image_dir, label_dir = split_paths(split)
        image_paths = sorted(image_dir.glob("*.jpg")) if image_dir.exists() else []
        empty_labels = 0
        objects = 0
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists() or not label_path.read_text(encoding="utf-8").strip():
                empty_labels += 1
                continue
            width, height = image_size(image_path)
            for idx, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    invalid.append({"split": split, "label": str(label_path), "line": idx, "reason": "bad_field_count"})
                    continue
                class_id, cx, cy, bw, bh = map(float, parts)
                objects += 1
                if bw <= 0 or bh <= 0:
                    invalid.append({"split": split, "label": str(label_path), "line": idx, "reason": "non_positive_size"})
                x1 = cx - bw / 2
                y1 = cy - bh / 2
                x2 = cx + bw / 2
                y2 = cy + bh / 2
                if min(cx, cy, bw, bh) < 0 or x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                    invalid.append({"split": split, "label": str(label_path), "line": idx, "reason": "out_of_bounds"})
                if math.isnan(class_id) or class_id < 0:
                    invalid.append({"split": split, "label": str(label_path), "line": idx, "reason": "invalid_class"})
        result["splits"][split] = {
            "images": len(image_paths),
            "objects": objects,
            "empty_labels": empty_labels,
        }
    result["invalid"] = invalid
    result["invalid_count"] = len(invalid)
    return result


def dataset_statistics(records: list[BoxRecord], splits: Iterable[str]) -> dict[str, object]:
    class_counts: dict[str, int] = {}
    size_counts = {"small": 0, "medium": 0, "large": 0}
    image_ids = set()
    for record in records:
        class_counts[record.class_name] = class_counts.get(record.class_name, 0) + 1
        size_counts[record.size_bucket] += 1
        image_ids.add((record.split, record.image_id))
    total_objects = len(records)
    total_images = 0
    for split in splits:
        _, _, image_dir, _ = split_paths(split)
        total_images += len(list(image_dir.glob("*.jpg"))) if image_dir.exists() else 0
    return {
        "total_images": total_images,
        "total_objects": total_objects,
        "average_objects_per_image": total_objects / total_images if total_images else 0,
        "class_distribution": dict(sorted(class_counts.items())),
        "size_distribution": size_counts,
        "average_bbox_width_px": float(np.mean([r.width for r in records])) if records else 0,
        "average_bbox_height_px": float(np.mean([r.height for r in records])) if records else 0,
        "average_bbox_area_px2": float(np.mean([r.area for r in records])) if records else 0,
        "splits": list(splits),
    }


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(
    ground_truths: list[BoxRecord],
    predictions: list[PredictionRecord],
    iou_threshold: float = 0.5,
    class_agnostic: bool = False,
) -> tuple[list[dict[str, object]], list[PredictionRecord], list[BoxRecord]]:
    gt_used: set[int] = set()
    matches: list[dict[str, object]] = []
    false_positives: list[PredictionRecord] = []
    sorted_preds = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    for prediction in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(ground_truths):
            if idx in gt_used:
                continue
            if not class_agnostic and prediction.class_id != gt.class_id:
                continue
            iou = iou_xyxy(prediction.to_xyxy(), gt.to_xyxy())
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            gt_used.add(best_idx)
            matches.append({"prediction": prediction, "ground_truth": ground_truths[best_idx], "iou": best_iou})
        else:
            false_positives.append(prediction)
    false_negatives = [gt for idx, gt in enumerate(ground_truths) if idx not in gt_used]
    return matches, false_positives, false_negatives


def metrics_from_matches(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def simple_ap50(matches: int, total_predictions: int, total_ground_truths: int) -> float:
    precision = matches / total_predictions if total_predictions else 0.0
    recall = matches / total_ground_truths if total_ground_truths else 0.0
    return precision * recall


def roi_boxes(strategy: str, width: int, height: int) -> list[tuple[float, float, float, float]]:
    if strategy == "horizon":
        return [(0, 0, width, height * 0.45)]
    if strategy == "center":
        return [(width * 0.2, height * 0.2, width * 0.8, height * 0.8)]
    if strategy == "uncertain":
        return [(0, 0, width, height * 0.35), (width * 0.15, height * 0.35, width * 0.85, height * 0.75)]
    if strategy == "density":
        return [(0, 0, width, height * 0.55), (width * 0.25, height * 0.55, width * 0.75, height)]
    raise ValueError(f"Unknown ROI strategy: {strategy}")


def box_center_inside_roi(record: BoxRecord, roi: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = roi
    return x1 <= record.center_x <= x2 and y1 <= record.center_y <= y2


def roi_area_ratio(rois: list[tuple[float, float, float, float]], width: int, height: int) -> float:
    area = sum(max(0.0, x2 - x1) * max(0.0, y2 - y1) for x1, y1, x2, y2 in rois)
    return min(1.0, area / (width * height))


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def records_to_jsonable(records: Iterable[object]) -> list[dict[str, object]]:
    return [asdict(record) for record in records]
