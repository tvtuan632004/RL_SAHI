from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    PredictionRecord,
    ensure_dirs,
    match_predictions,
    metrics_from_matches,
    now_ms,
    read_yolo_labels,
    records_to_jsonable,
    save_json,
    simple_ap50,
    split_paths,
)


BASE_DIR = Path(__file__).resolve().parent
BASELINE_DIR = BASE_DIR / "baseline"
OUTPUT_DIR = BASE_DIR / "outputs"


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def run_yolo_inference(model_name: str, split: str, conf: float, imgsz: int, sample_limit: int | None):
    local_ultralytics_dir = OUTPUT_DIR / ".ultralytics"
    local_ultralytics_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_ultralytics_dir))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before running baseline inference.") from exc

    _, _, image_dir, _ = split_paths(split)
    image_paths = sorted(image_dir.glob("*.jpg"))
    if sample_limit:
        image_paths = image_paths[:sample_limit]

    model_ref = Path(model_name)
    if model_ref.suffix == ".pt" and len(model_ref.parts) == 1:
        model_ref = BASELINE_DIR / model_name
    model = YOLO(str(model_ref))
    predictions_by_image: dict[str, list[PredictionRecord]] = defaultdict(list)
    latency_ms: dict[str, float] = {}

    for idx, image_path in enumerate(image_paths, start=1):
        start = now_ms()
        results = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)
        latency_ms[image_path.stem] = now_ms() - start
        if not results:
            continue
        result = results[0]
        height, width = result.orig_shape
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, scores, classes):
            predictions_by_image[image_path.stem].append(
                PredictionRecord(
                    image_id=image_path.stem,
                    class_id=int(class_id),
                    confidence=float(score),
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    image_width=int(width),
                    image_height=int(height),
                )
            )
        if idx % 100 == 0:
            print(f"Inferenced {idx}/{len(image_paths)} images")

    return predictions_by_image, latency_ms, [path.stem for path in image_paths]


def evaluate(
    split: str,
    predictions_by_image: dict[str, list[PredictionRecord]],
    image_ids: list[str],
    class_agnostic: bool,
):
    gt_all = read_yolo_labels(split)
    gt_by_image = defaultdict(list)
    for gt in gt_all:
        if gt.image_id in image_ids:
            gt_by_image[gt.image_id].append(gt)

    all_matches = []
    all_fp = []
    all_fn = []
    by_size = {
        "small": {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "gt": 0},
        "medium": {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "gt": 0},
        "large": {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "gt": 0},
    }

    for image_id in image_ids:
        matches, fp, fn = match_predictions(
            gt_by_image[image_id],
            predictions_by_image.get(image_id, []),
            iou_threshold=0.5,
            class_agnostic=class_agnostic,
        )
        all_matches.extend(matches)
        all_fp.extend(fp)
        all_fn.extend(fn)
        for match in matches:
            bucket = match["ground_truth"].size_bucket
            by_size[bucket]["tp"] += 1
        for pred in fp:
            by_size[pred.size_bucket]["fp"] += 1
        for gt in fn:
            by_size[gt.size_bucket]["fn"] += 1
        for pred in predictions_by_image.get(image_id, []):
            by_size[pred.size_bucket]["pred"] += 1
        for gt in gt_by_image[image_id]:
            by_size[gt.size_bucket]["gt"] += 1

    summary = metrics_from_matches(len(all_matches), len(all_fp), len(all_fn))
    summary["mAP50_proxy"] = simple_ap50(len(all_matches), len(all_matches) + len(all_fp), len(all_matches) + len(all_fn))
    size_metrics = {}
    for bucket, values in by_size.items():
        metrics = metrics_from_matches(values["tp"], values["fp"], values["fn"])
        metrics["AP50_proxy"] = simple_ap50(values["tp"], values["pred"], values["gt"])
        size_metrics[bucket] = metrics

    return summary, size_metrics, all_matches, all_fp, all_fn


def plot_small_object_metrics(size_metrics: dict[str, dict[str, float]]) -> None:
    buckets = ["small", "medium", "large"]
    precision = [size_metrics[b]["precision"] for b in buckets]
    recall = [size_metrics[b]["recall"] for b in buckets]
    ap50 = [size_metrics[b]["AP50_proxy"] for b in buckets]

    x = np.arange(len(buckets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, precision, width, label="Precision", color="#3b82f6")
    ax.bar(x, recall, width, label="Recall", color="#10b981")
    ax.bar(x + width, ap50, width, label="AP50 proxy", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLO Baseline Performance by Object Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "small_object_metrics.png")
    plt.close(fig)


def plot_failure_heatmap(false_negatives) -> None:
    x = [gt.norm_center_x for gt in false_negatives]
    y = [gt.norm_center_y for gt in false_negatives]

    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.hist2d(x, y, bins=60, range=[[0, 1], [0, 1]], cmap="inferno")
    ax.invert_yaxis()
    ax.set_title("YOLO Missed Detection Heatmap")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    fig.colorbar(heat[3], ax=ax, label="Missed objects")
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "failure_heatmap.png")
    plt.close(fig)


def plot_confidence_analysis(matches, false_positives, false_negatives) -> None:
    tp_conf = [m["prediction"].confidence for m in matches]
    fp_conf = [p.confidence for p in false_positives]
    all_pred = [m["prediction"] for m in matches] + false_positives

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(tp_conf, bins=30, alpha=0.75, label="TP", color="#10b981")
    axes[0].hist(fp_conf, bins=30, alpha=0.65, label="FP", color="#ef4444")
    axes[0].set_title("Confidence Distribution")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Prediction count")
    axes[0].legend()

    axes[1].scatter([p.area for p in all_pred], [p.confidence for p in all_pred], s=8, alpha=0.35, color="#3b82f6")
    axes[1].set_xscale("log")
    axes[1].set_title("Confidence vs Predicted Object Size")
    axes[1].set_xlabel("Predicted area (px^2, log)")
    axes[1].set_ylabel("Confidence")

    buckets = ["small", "medium", "large"]
    miss_counts = {bucket: 0 for bucket in buckets}
    gt_counts = {bucket: 0 for bucket in buckets}
    for m in matches:
        gt_counts[m["ground_truth"].size_bucket] += 1
    for gt in false_negatives:
        miss_counts[gt.size_bucket] += 1
        gt_counts[gt.size_bucket] += 1
    miss_rate = [miss_counts[b] / gt_counts[b] if gt_counts[b] else 0 for b in buckets]
    axes[2].bar(buckets, miss_rate, color=["#ef4444", "#f59e0b", "#22c55e"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Miss Rate by Object Size")
    axes[2].set_ylabel("Miss rate")

    fig.suptitle("YOLO Confidence and Failure Behavior")
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "confidence_analysis.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO validation baseline and failure analysis.")
    parser.add_argument("--model", default="yolo11s.pt", help="Ultralytics YOLO model path/name.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of images for quick experiments.")
    parser.add_argument(
        "--class-agnostic",
        action="store_true",
        help="Match predictions to ground truth by IoU only, useful for COCO-pretrained YOLO on VisDrone.",
    )
    args = parser.parse_args()

    ensure_dirs(BASELINE_DIR, OUTPUT_DIR)
    setup_plot_style()

    predictions_by_image, latency_ms, image_ids = run_yolo_inference(
        model_name=args.model,
        split=args.split,
        conf=args.conf,
        imgsz=args.imgsz,
        sample_limit=args.sample_limit,
    )
    summary, size_metrics, matches, false_positives, false_negatives = evaluate(
        args.split,
        predictions_by_image,
        image_ids,
        class_agnostic=args.class_agnostic,
    )

    prediction_json = {
        image_id: records_to_jsonable(predictions)
        for image_id, predictions in predictions_by_image.items()
    }
    save_json(prediction_json, OUTPUT_DIR / "baseline_predictions.json")
    save_json(latency_ms, OUTPUT_DIR / "baseline_latency_ms.json")
    save_json(
        {"overall": summary, "by_size": size_metrics, "class_agnostic": args.class_agnostic},
        OUTPUT_DIR / "baseline_metrics.json",
    )
    save_json(records_to_jsonable(false_positives), OUTPUT_DIR / "baseline_false_positives.json")
    save_json(records_to_jsonable(false_negatives), OUTPUT_DIR / "baseline_false_negatives.json")

    plot_small_object_metrics(size_metrics)
    plot_failure_heatmap(false_negatives)
    plot_confidence_analysis(matches, false_positives, false_negatives)

    mean_latency = float(np.mean(list(latency_ms.values()))) if latency_ms else 0.0
    print(f"Baseline complete on {len(image_ids)} images. Mean latency: {mean_latency:.2f} ms.")
    print(f"Metrics: precision={summary['precision']:.4f}, recall={summary['recall']:.4f}, mAP50_proxy={summary['mAP50_proxy']:.4f}")


if __name__ == "__main__":
    main()
