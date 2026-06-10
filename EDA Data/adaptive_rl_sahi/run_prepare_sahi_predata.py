from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from utils import (
    BoxRecord,
    PredictionRecord,
    box_center_inside_roi,
    ensure_dirs,
    load_json,
    read_yolo_labels,
    roi_area_ratio,
    roi_boxes,
    save_json,
    split_paths,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ROI_STRATEGIES = ("horizon", "center", "uncertain", "density")


def box_to_dict(record: BoxRecord) -> dict[str, object]:
    return {
        "class_id": record.class_id,
        "class_name": record.class_name,
        "bbox_xyxy": [record.x1, record.y1, record.x2, record.y2],
        "bbox_xywh": [record.x1, record.y1, record.width, record.height],
        "center_xy": [record.center_x, record.center_y],
        "size_bucket": record.size_bucket,
        "area": record.area,
    }


def prediction_to_dict(record: PredictionRecord) -> dict[str, object]:
    return {
        "class_id": record.class_id,
        "confidence": record.confidence,
        "bbox_xyxy": [record.x1, record.y1, record.x2, record.y2],
        "bbox_xywh": [record.x1, record.y1, record.width, record.height],
        "size_bucket": record.size_bucket,
        "area": record.area,
    }


def dict_to_box_record(item: dict) -> BoxRecord:
    return BoxRecord(
        image_id=item["image_id"],
        split=item.get("split", "test"),
        class_id=int(item["class_id"]),
        class_name=item.get("class_name", f"class_{item['class_id']}"),
        x1=float(item["x1"]),
        y1=float(item["y1"]),
        x2=float(item["x2"]),
        y2=float(item["y2"]),
        width=float(item["width"]),
        height=float(item["height"]),
        image_width=int(item["image_width"]),
        image_height=int(item["image_height"]),
    )


def dict_to_prediction_record(image_id: str, item: dict) -> PredictionRecord:
    return PredictionRecord(
        image_id=image_id,
        class_id=int(item["class_id"]),
        confidence=float(item["confidence"]),
        x1=float(item["x1"]),
        y1=float(item["y1"]),
        x2=float(item["x2"]),
        y2=float(item["y2"]),
        image_width=int(item["image_width"]),
        image_height=int(item["image_height"]),
    )


def load_predictions(split: str) -> dict[str, list[PredictionRecord]]:
    path = OUTPUT_DIR / "baseline_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_baseline.py --split {split} first.")

    raw = load_json(path)
    predictions: dict[str, list[PredictionRecord]] = {}
    for image_id, items in raw.items():
        predictions[image_id] = [dict_to_prediction_record(image_id, item) for item in items]
    return predictions


def load_missed(split: str) -> dict[str, list[BoxRecord]]:
    path = OUTPUT_DIR / "baseline_false_negatives.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run run_baseline.py --split {split} first.")

    missed_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for item in load_json(path):
        record = dict_to_box_record(item)
        if record.split == split:
            missed_by_image[record.image_id].append(record)
    return missed_by_image


def summarize_counts(records: list[BoxRecord]) -> dict[str, object]:
    by_class = Counter(record.class_name for record in records)
    by_size = Counter(record.size_bucket for record in records)
    return {
        "total": len(records),
        "by_class": dict(sorted(by_class.items())),
        "by_size": {key: by_size.get(key, 0) for key in ("small", "medium", "large")},
    }


def build_roi_candidates(
    width: int,
    height: int,
    small_objects: list[BoxRecord],
    missed_objects: list[BoxRecord],
) -> list[dict[str, object]]:
    candidates = []
    for strategy in ROI_STRATEGIES:
        rois = roi_boxes(strategy, width, height)
        small_covered = sum(
            1 for record in small_objects if any(box_center_inside_roi(record, roi) for roi in rois)
        )
        missed_covered = sum(
            1 for record in missed_objects if any(box_center_inside_roi(record, roi) for roi in rois)
        )
        area_ratio = roi_area_ratio(rois, width, height)
        small_coverage = small_covered / len(small_objects) if small_objects else 0.0
        missed_coverage = missed_covered / len(missed_objects) if missed_objects else 0.0
        candidates.append(
            {
                "strategy": strategy,
                "boxes_xyxy": [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in rois],
                "roi_area_ratio": area_ratio,
                "small_objects_covered": small_covered,
                "small_object_coverage": small_coverage,
                "missed_objects_covered": missed_covered,
                "missed_object_coverage": missed_coverage,
                "efficiency_score": missed_coverage / area_ratio if area_ratio else 0.0,
            }
        )
    return sorted(candidates, key=lambda item: item["efficiency_score"], reverse=True)


def build_image_record(
    image_path: Path,
    gt_records: list[BoxRecord],
    predictions: list[PredictionRecord],
    missed: list[BoxRecord],
) -> dict[str, object]:
    width, height = gt_records[0].image_width, gt_records[0].image_height
    small_objects = [record for record in gt_records if record.size_bucket == "small"]
    missed_small = [record for record in missed if record.size_bucket == "small"]
    detected = len(gt_records) - len(missed)
    recall = detected / len(gt_records) if gt_records else 0.0
    miss_rate = len(missed) / len(gt_records) if gt_records else 0.0
    candidates = build_roi_candidates(width, height, small_objects, missed, )
    best_candidate = candidates[0]["strategy"] if candidates else None

    return {
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "summary": {
            "total_labels": len(gt_records),
            "detected_labels": detected,
            "missed_labels": len(missed),
            "recall": recall,
            "miss_rate": miss_rate,
            "small_labels": len(small_objects),
            "missed_small_labels": len(missed_small),
            "yolo_predictions": len(predictions),
            "best_roi_strategy_by_missed_efficiency": best_candidate,
        },
        "roi_candidates": candidates,
        "ground_truth": [box_to_dict(record) for record in gt_records],
        "yolo_predictions": [prediction_to_dict(record) for record in predictions],
        "missed_objects": [box_to_dict(record) for record in missed],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare per-image pre-data for SAHI and adaptive ROI experiments.")
    parser.add_argument("--split", default="test", help="Dataset split to prepare.")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output JSON name. Default: sahi_<split>_predata.json",
    )
    args = parser.parse_args()

    ensure_dirs(OUTPUT_DIR)
    _, _, image_dir, _ = split_paths(args.split)
    image_paths = {path.stem: path for path in sorted(image_dir.glob("*.jpg"))}
    gt_by_image: dict[str, list[BoxRecord]] = defaultdict(list)
    for record in read_yolo_labels(args.split):
        gt_by_image[record.image_id].append(record)

    predictions_by_image = load_predictions(args.split)
    missed_by_image = load_missed(args.split)

    images = []
    all_gt = []
    all_missed = []
    roi_strategy_wins = Counter()
    for image_id, gt_records in sorted(gt_by_image.items()):
        image_path = image_paths.get(image_id)
        if image_path is None:
            continue
        predictions = predictions_by_image.get(image_id, [])
        missed = missed_by_image.get(image_id, [])
        image_record = build_image_record(image_path, gt_records, predictions, missed)
        images.append(image_record)
        all_gt.extend(gt_records)
        all_missed.extend(missed)
        best_strategy = image_record["summary"]["best_roi_strategy_by_missed_efficiency"]
        if best_strategy:
            roi_strategy_wins[best_strategy] += 1

    total_labels = len(all_gt)
    missed_labels = len(all_missed)
    summary = {
        "split": args.split,
        "total_images": len(images),
        "total_labels": total_labels,
        "missed_labels": missed_labels,
        "detected_labels": total_labels - missed_labels,
        "recall": (total_labels - missed_labels) / total_labels if total_labels else 0.0,
        "miss_rate": missed_labels / total_labels if total_labels else 0.0,
        "ground_truth": summarize_counts(all_gt),
        "missed": summarize_counts(all_missed),
        "roi_strategy_wins": dict(sorted(roi_strategy_wins.items())),
    }

    output = {
        "description": "Per-image SAHI pre-data built from YOLO full-image baseline outputs.",
        "notes": [
            "Use roi_candidates as heuristic regions for SAHI slicing experiments.",
            "The current missed_objects field comes from YOLO baseline false negatives.",
            "This file does not run SAHI yet; it prepares comparable inputs for future SAHI runs.",
        ],
        "summary": summary,
        "images": images,
    }

    output_name = args.output_name or f"sahi_{args.split}_predata.json"
    output_path = OUTPUT_DIR / output_name
    save_json(output, output_path)
    save_json(summary, OUTPUT_DIR / f"sahi_{args.split}_predata_summary.json")

    print(f"SAHI pre-data saved: {output_path}")
    print(
        f"Images={summary['total_images']}, labels={summary['total_labels']}, "
        f"missed={summary['missed_labels']}, miss_rate={summary['miss_rate']:.4f}"
    )


if __name__ == "__main__":
    main()

