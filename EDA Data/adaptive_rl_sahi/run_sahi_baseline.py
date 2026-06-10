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
    load_json,
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


def resolve_model_path(model_name: str) -> Path:
    model_ref = Path(model_name)
    if model_ref.suffix == ".pt" and len(model_ref.parts) == 1:
        return BASELINE_DIR / model_name
    return model_ref


def build_sahi_model(model_path: Path, conf: float, device: str):
    local_ultralytics_dir = OUTPUT_DIR / ".ultralytics"
    local_ultralytics_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_ultralytics_dir))

    try:
        from sahi import AutoDetectionModel
    except ImportError as exc:
        raise RuntimeError("Install sahi before running SAHI baseline.") from exc

    return AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(model_path),
        confidence_threshold=conf,
        device=device,
    )


def sahi_prediction_to_record(image_id: str, prediction, width: int, height: int) -> PredictionRecord:
    x1, y1, x2, y2 = prediction.bbox.to_xyxy()
    return PredictionRecord(
        image_id=image_id,
        class_id=int(prediction.category.id),
        confidence=float(prediction.score.value),
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        image_width=width,
        image_height=height,
    )


def run_sahi_inference(args: argparse.Namespace):
    try:
        from sahi.predict import get_sliced_prediction
    except ImportError as exc:
        raise RuntimeError("Install sahi before running SAHI baseline.") from exc

    _, _, image_dir, _ = split_paths(args.split)
    image_paths = sorted(image_dir.glob("*.jpg"))
    if args.sample_limit:
        image_paths = image_paths[: args.sample_limit]

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    detection_model = build_sahi_model(model_path, args.conf, args.device)
    predictions_by_image: dict[str, list[PredictionRecord]] = defaultdict(list)
    latency_ms: dict[str, float] = {}

    for idx, image_path in enumerate(image_paths, start=1):
        start = now_ms()
        result = get_sliced_prediction(
            str(image_path),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            postprocess_type=args.postprocess_type,
            postprocess_match_metric=args.postprocess_match_metric,
            postprocess_match_threshold=args.postprocess_match_threshold,
            verbose=0,
        )
        latency_ms[image_path.stem] = now_ms() - start
        for prediction in result.object_prediction_list:
            predictions_by_image[image_path.stem].append(
                sahi_prediction_to_record(
                    image_id=image_path.stem,
                    prediction=prediction,
                    width=result.image_width,
                    height=result.image_height,
                )
            )
        if idx % args.log_every == 0:
            print(f"SAHI inferenced {idx}/{len(image_paths)} images")

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
            by_size[match["ground_truth"].size_bucket]["tp"] += 1
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


def plot_sahi_small_metrics(size_metrics: dict[str, dict[str, float]]) -> None:
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
    ax.set_title("SAHI Baseline Performance by Object Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "sahi_small_object_metrics.png")
    plt.close(fig)


def plot_sahi_failure_heatmap(false_negatives) -> None:
    x = [gt.norm_center_x for gt in false_negatives]
    y = [gt.norm_center_y for gt in false_negatives]

    fig, ax = plt.subplots(figsize=(6, 5))
    heat = ax.hist2d(x, y, bins=70, range=[[0, 1], [0, 1]], cmap="inferno")
    ax.invert_yaxis()
    ax.set_title("SAHI Missed Detection Heatmap")
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    fig.colorbar(heat[3], ax=ax, label="Missed objects")
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "sahi_failure_heatmap.png")
    plt.close(fig)


def plot_sahi_vs_yolo(sahi_metrics: dict[str, object]) -> None:
    yolo_path = OUTPUT_DIR / "baseline_metrics.json"
    if not yolo_path.exists():
        return
    yolo_metrics = load_json(yolo_path)

    labels = ["precision", "recall", "mAP50_proxy"]
    yolo_values = [yolo_metrics["overall"][label] for label in labels]
    sahi_values = [sahi_metrics["overall"][label] for label in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, yolo_values, width, label="YOLO full-image", color="#64748b")
    ax.bar(x + width / 2, sahi_values, width, label="SAHI slicing", color="#8b5cf6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLO Full-Image vs SAHI Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "sahi_vs_yolo_comparison.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAHI sliced inference and compare with YOLO baseline.")
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO model path/name.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="cpu", help="SAHI/Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--slice-height", type=int, default=512, help="SAHI slice height.")
    parser.add_argument("--slice-width", type=int, default=512, help="SAHI slice width.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Slice overlap ratio.")
    parser.add_argument("--postprocess-type", default="GREEDYNMM", help="SAHI postprocess type.")
    parser.add_argument("--postprocess-match-metric", default="IOS", help="SAHI postprocess match metric.")
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.5, help="SAHI postprocess threshold.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit image count for smoke tests.")
    parser.add_argument("--log-every", type=int, default=50, help="Progress logging interval.")
    parser.add_argument(
        "--strict-class",
        action="store_true",
        help="Require class IDs to match. By default, evaluation is class-agnostic for COCO-pretrained YOLO.",
    )
    args = parser.parse_args()

    ensure_dirs(BASELINE_DIR, OUTPUT_DIR)
    setup_plot_style()

    predictions_by_image, latency_ms, image_ids = run_sahi_inference(args)
    summary, size_metrics, matches, false_positives, false_negatives = evaluate(
        split=args.split,
        predictions_by_image=predictions_by_image,
        image_ids=image_ids,
        class_agnostic=not args.strict_class,
    )

    prediction_json = {
        image_id: records_to_jsonable(predictions)
        for image_id, predictions in predictions_by_image.items()
    }
    metrics = {
        "overall": summary,
        "by_size": size_metrics,
        "class_agnostic": not args.strict_class,
        "sahi_config": {
            "model": str(resolve_model_path(args.model)),
            "split": args.split,
            "conf": args.conf,
            "device": args.device,
            "slice_height": args.slice_height,
            "slice_width": args.slice_width,
            "overlap": args.overlap,
            "postprocess_type": args.postprocess_type,
            "postprocess_match_metric": args.postprocess_match_metric,
            "postprocess_match_threshold": args.postprocess_match_threshold,
            "sample_limit": args.sample_limit,
        },
    }

    save_json(prediction_json, OUTPUT_DIR / "sahi_predictions.json")
    save_json(latency_ms, OUTPUT_DIR / "sahi_latency_ms.json")
    save_json(metrics, OUTPUT_DIR / "sahi_metrics.json")
    save_json(records_to_jsonable(false_positives), OUTPUT_DIR / "sahi_false_positives.json")
    save_json(records_to_jsonable(false_negatives), OUTPUT_DIR / "sahi_false_negatives.json")

    plot_sahi_small_metrics(size_metrics)
    plot_sahi_failure_heatmap(false_negatives)
    plot_sahi_vs_yolo(metrics)

    mean_latency = float(np.mean(list(latency_ms.values()))) if latency_ms else 0.0
    print(f"SAHI baseline complete on {len(image_ids)} images. Mean latency: {mean_latency:.2f} ms.")
    print(
        f"Metrics: precision={summary['precision']:.4f}, "
        f"recall={summary['recall']:.4f}, mAP50_proxy={summary['mAP50_proxy']:.4f}"
    )


if __name__ == "__main__":
    main()

