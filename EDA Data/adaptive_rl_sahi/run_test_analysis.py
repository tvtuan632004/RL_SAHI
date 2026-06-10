from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import BoxRecord, ensure_dirs, load_json, read_yolo_labels, save_json


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
VIS_DIR = BASE_DIR / "visualization"


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


def dict_to_box_record(item: dict) -> BoxRecord:
    return BoxRecord(
        image_id=item["image_id"],
        split=item.get("split", "val"),
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


def load_false_negatives(split: str) -> list[BoxRecord]:
    path = OUTPUT_DIR / "baseline_false_negatives.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_baseline.py on split '{split}' before test analysis."
        )
    records = [dict_to_box_record(item) for item in load_json(path)]
    return [record for record in records if record.split == split]


def count_by(records: list[BoxRecord], attr: str) -> dict[str, int]:
    return dict(Counter(str(getattr(record, attr)) for record in records))


def grouped_summary(gt_records: list[BoxRecord], miss_records: list[BoxRecord], attr: str) -> dict[str, dict[str, float]]:
    gt_counts = Counter(str(getattr(record, attr)) for record in gt_records)
    miss_counts = Counter(str(getattr(record, attr)) for record in miss_records)
    keys = sorted(set(gt_counts) | set(miss_counts))
    summary = {}
    for key in keys:
        total = gt_counts[key]
        missed = miss_counts[key]
        detected = total - missed
        summary[key] = {
            "total_labels": total,
            "detected": detected,
            "missed": missed,
            "recall": detected / total if total else 0.0,
            "miss_rate": missed / total if total else 0.0,
        }
    return summary


def top_missed_images(gt_records: list[BoxRecord], miss_records: list[BoxRecord], limit: int) -> list[dict[str, float]]:
    gt_by_image = Counter(record.image_id for record in gt_records)
    miss_by_image = Counter(record.image_id for record in miss_records)
    rows = []
    for image_id, missed in miss_by_image.most_common(limit):
        total = gt_by_image[image_id]
        rows.append(
            {
                "image_id": image_id,
                "total_labels": total,
                "missed": missed,
                "detected": total - missed,
                "miss_rate": missed / total if total else 0.0,
            }
        )
    return rows


def plot_heatmaps(gt_records: list[BoxRecord], miss_records: list[BoxRecord], split: str) -> None:
    small_gt = [record for record in gt_records if record.size_bucket == "small"]
    small_miss = [record for record in miss_records if record.size_bucket == "small"]
    panels = [
        ("All Ground Truth Labels", gt_records, "viridis"),
        ("All Missed Labels", miss_records, "inferno"),
        ("Small Ground Truth Labels", small_gt, "plasma"),
        ("Small Missed Labels", small_miss, "magma"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (title, records, cmap) in zip(axes.ravel(), panels):
        x = [record.norm_center_x for record in records]
        y = [record.norm_center_y for record in records]
        heat = ax.hist2d(x, y, bins=70, range=[[0, 1], [0, 1]], cmap=cmap)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        fig.colorbar(heat[3], ax=ax, label="Object count")

    fig.suptitle(f"Ground Truth vs YOLO Miss Heatmaps ({split})")
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"{split}_gt_vs_missed_heatmaps.png")
    plt.close(fig)


def plot_label_miss_bars(class_summary: dict, size_summary: dict, split: str) -> None:
    class_names = list(class_summary.keys())
    class_total = [class_summary[name]["total_labels"] for name in class_names]
    class_missed = [class_summary[name]["missed"] for name in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(class_names))
    width = 0.38
    axes[0].bar(x - width / 2, class_total, width, label="Total labels", color="#3b82f6")
    axes[0].bar(x + width / 2, class_missed, width, label="Missed by YOLO", color="#ef4444")
    axes[0].set_title("Labels vs Missed Labels by Class")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=35, ha="right")
    axes[0].set_ylabel("Object count")
    axes[0].legend()

    sizes = ["small", "medium", "large"]
    total = [size_summary[size]["total_labels"] for size in sizes]
    missed = [size_summary[size]["missed"] for size in sizes]
    miss_rate = [size_summary[size]["miss_rate"] for size in sizes]
    x2 = np.arange(len(sizes))
    axes[1].bar(x2 - width / 2, total, width, label="Total labels", color="#10b981")
    axes[1].bar(x2 + width / 2, missed, width, label="Missed by YOLO", color="#f97316")
    axes[1].set_title("Labels vs Missed Labels by Size")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(sizes)
    axes[1].set_ylabel("Object count")
    rate_axis = axes[1].twinx()
    rate_axis.plot(x2, miss_rate, marker="o", color="#111827", label="Miss rate")
    rate_axis.set_ylim(0, 1)
    rate_axis.set_ylabel("Miss rate")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = rate_axis.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.suptitle(f"YOLO Label Coverage Summary ({split})")
    fig.tight_layout()
    fig.savefig(VIS_DIR / f"{split}_label_miss_summary.png")
    plt.close(fig)


def write_markdown_report(report: dict, split: str) -> None:
    lines = [
        f"# Test Analysis Report ({split})",
        "",
        f"- Total images: {report['total_images']}",
        f"- Total ground-truth labels: {report['total_labels']}",
        f"- Detected labels: {report['detected_labels']}",
        f"- Missed labels: {report['missed_labels']}",
        f"- Recall: {report['recall']:.4f}",
        f"- Miss rate: {report['miss_rate']:.4f}",
        "",
        "## By Size",
    ]
    for size, values in report["by_size"].items():
        lines.append(
            f"- {size}: total={values['total_labels']}, missed={values['missed']}, "
            f"recall={values['recall']:.4f}, miss_rate={values['miss_rate']:.4f}"
        )
    lines.extend(["", "## Top Missed Images"])
    for row in report["top_missed_images"]:
        lines.append(
            f"- {row['image_id']}: missed={row['missed']}/{row['total_labels']} "
            f"({row['miss_rate']:.2%})"
        )
    (OUTPUT_DIR / f"{split}_test_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize labels, YOLO misses, and heatmaps for a dataset split.")
    parser.add_argument("--split", default="val", help="Split to analyze. Must match the latest baseline output.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of high-miss images to list.")
    args = parser.parse_args()

    ensure_dirs(OUTPUT_DIR, VIS_DIR)
    setup_plot_style()

    gt_records = read_yolo_labels(args.split)
    miss_records = load_false_negatives(args.split)
    total_labels = len(gt_records)
    missed = len(miss_records)
    detected = total_labels - missed

    report = {
        "split": args.split,
        "total_images": len(set(record.image_id for record in gt_records)),
        "total_labels": total_labels,
        "detected_labels": detected,
        "missed_labels": missed,
        "recall": detected / total_labels if total_labels else 0.0,
        "miss_rate": missed / total_labels if total_labels else 0.0,
        "by_class": grouped_summary(gt_records, miss_records, "class_name"),
        "by_size": grouped_summary(gt_records, miss_records, "size_bucket"),
        "top_missed_images": top_missed_images(gt_records, miss_records, args.top_k),
    }

    save_json(report, OUTPUT_DIR / f"{args.split}_test_analysis.json")
    plot_heatmaps(gt_records, miss_records, args.split)
    plot_label_miss_bars(report["by_class"], report["by_size"], args.split)
    write_markdown_report(report, args.split)

    print(f"Test analysis complete for split={args.split}")
    print(f"Total labels={total_labels}, missed={missed}, recall={report['recall']:.4f}, miss_rate={report['miss_rate']:.4f}")


if __name__ == "__main__":
    main()
