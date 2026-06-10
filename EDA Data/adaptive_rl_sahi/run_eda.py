from __future__ import annotations

from collections import Counter, defaultdict
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    OUTPUT_ROOT,
    dataset_statistics,
    ensure_dirs,
    read_all_yolo_labels,
    save_json,
    validate_yolo_dataset,
)


BASE_DIR = Path(__file__).resolve().parent
EDA_DIR = BASE_DIR / "eda"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_SPLITS = ("train", "val", "test")


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


def plot_bbox_distribution(records) -> None:
    widths = [r.width for r in records]
    heights = [r.height for r in records]
    areas = [r.area for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(widths, bins=60, color="#3b82f6", alpha=0.85)
    axes[0].set_title("BBox Width Distribution")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Object count")

    axes[1].hist(heights, bins=60, color="#10b981", alpha=0.85)
    axes[1].set_title("BBox Height Distribution")
    axes[1].set_xlabel("Height (px)")

    axes[2].hist(areas, bins=60, color="#f59e0b", alpha=0.85)
    axes[2].set_title("BBox Area Distribution")
    axes[2].set_xlabel("Area (px^2)")

    fig.suptitle("VisDrone Object Bounding Box Distribution")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "bbox_distribution.png")
    plt.close(fig)


def plot_object_size_ratio(records) -> dict[str, int]:
    counts = Counter(r.size_bucket for r in records)
    labels = ["small", "medium", "large"]
    values = [counts[label] for label in labels]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#ef4444", "#f59e0b", "#22c55e"])
    axes[0].set_title("COCO-Style Object Size Ratio")

    axes[1].axis("off")
    total = sum(values)
    table_data = [[label, values[idx], f"{values[idx] / total * 100:.2f}%"] for idx, label in enumerate(labels)]
    table = axes[1].table(cellText=table_data, colLabels=["Size", "Objects", "Ratio"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    axes[1].set_title("Size Statistics Table")

    fig.tight_layout()
    fig.savefig(EDA_DIR / "object_size_ratio.png")
    plt.close(fig)
    return dict(counts)


def plot_spatial_heatmap(records) -> None:
    all_x = [r.norm_center_x for r in records]
    all_y = [r.norm_center_y for r in records]
    small = [r for r in records if r.size_bucket == "small"]
    small_x = [r.norm_center_x for r in small]
    small_y = [r.norm_center_y for r in small]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    h0 = axes[0].hist2d(all_x, all_y, bins=60, range=[[0, 1], [0, 1]], cmap="viridis")
    axes[0].invert_yaxis()
    axes[0].set_title("All Object Centers")
    axes[0].set_xlabel("Normalized X")
    axes[0].set_ylabel("Normalized Y")
    fig.colorbar(h0[3], ax=axes[0], label="Count")

    h1 = axes[1].hist2d(small_x, small_y, bins=60, range=[[0, 1], [0, 1]], cmap="magma")
    axes[1].invert_yaxis()
    axes[1].set_title("Small Object Centers")
    axes[1].set_xlabel("Normalized X")
    fig.colorbar(h1[3], ax=axes[1], label="Count")

    fig.suptitle("Spatial Heatmap Analysis")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "spatial_heatmap.png")
    plt.close(fig)


def plot_density(records) -> dict[str, dict[str, int]]:
    per_image = defaultdict(int)
    small_per_image = defaultdict(int)
    for record in records:
        key = f"{record.split}/{record.image_id}"
        per_image[key] += 1
        if record.size_bucket == "small":
            small_per_image[key] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(list(per_image.values()), bins=50, color="#6366f1", alpha=0.85)
    axes[0].set_title("Objects per Image")
    axes[0].set_xlabel("Object count")
    axes[0].set_ylabel("Image count")

    axes[1].hist(list(small_per_image.values()), bins=50, color="#ec4899", alpha=0.85)
    axes[1].set_title("Small Objects per Image")
    axes[1].set_xlabel("Small object count")

    fig.suptitle("Density Analysis")
    fig.tight_layout()
    fig.savefig(EDA_DIR / "density_analysis.png")
    plt.close(fig)
    return {
        "objects_per_image": dict(per_image),
        "small_objects_per_image": dict(small_per_image),
    }


def plot_border_analysis(records) -> dict[str, float]:
    distances = []
    near_border = 0
    threshold = 32
    for record in records:
        distance = min(record.x1, record.y1, record.image_width - record.x2, record.image_height - record.y2)
        distances.append(distance)
        if distance <= threshold:
            near_border += 1

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances, bins=60, color="#14b8a6", alpha=0.85)
    ax.axvline(threshold, color="#ef4444", linestyle="--", label="32 px border threshold")
    ax.set_title("Distance-to-Border Distribution")
    ax.set_xlabel("Minimum distance to image border (px)")
    ax.set_ylabel("Object count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(EDA_DIR / "border_analysis.png")
    plt.close(fig)

    return {
        "near_border_threshold_px": threshold,
        "near_border_objects": near_border,
        "near_border_ratio": near_border / len(records) if records else 0,
        "mean_distance_to_border_px": float(np.mean(distances)) if distances else 0,
    }


def write_insights(stats: dict, size_counts: dict, border_stats: dict) -> None:
    total = max(1, stats["total_objects"])
    small_ratio = size_counts.get("small", 0) / total
    text = [
        "# Research Insights Summary",
        "",
        f"- Dataset contains {stats['total_images']} images and {stats['total_objects']} labeled objects.",
        f"- Average density is {stats['average_objects_per_image']:.2f} objects per image.",
        f"- Small objects represent {small_ratio * 100:.2f}% of all annotations using COCO thresholds.",
        f"- {border_stats['near_border_ratio'] * 100:.2f}% of objects are within {border_stats['near_border_threshold_px']} px of an image border.",
        "- Use spatial_heatmap.png to identify where small-object slicing should be concentrated.",
        "- Use density_analysis.png to identify images where full-frame YOLO is likely to be stressed by clutter.",
        "",
        "RL is intentionally excluded in this phase.",
    ]
    (OUTPUT_DIR / "research_insights_summary.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dataset EDA for VisDrone YOLO labels.")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Dataset splits to analyze.")
    args = parser.parse_args()
    splits = tuple(args.splits)

    ensure_dirs(EDA_DIR, OUTPUT_DIR, OUTPUT_ROOT)
    setup_plot_style()

    validation = validate_yolo_dataset(splits)
    records = read_all_yolo_labels(splits)
    stats = dataset_statistics(records, splits)

    save_json(validation, OUTPUT_DIR / "dataset_validation.json")
    save_json(stats, OUTPUT_DIR / "dataset_stats.json")

    plot_bbox_distribution(records)
    size_counts = plot_object_size_ratio(records)
    plot_spatial_heatmap(records)
    density = plot_density(records)
    border_stats = plot_border_analysis(records)

    save_json({"size_counts": size_counts, "border": border_stats}, OUTPUT_DIR / "eda_statistics.json")
    save_json(density, OUTPUT_DIR / "density_per_image.json")
    write_insights(stats, size_counts, border_stats)

    print(f"EDA complete. Outputs saved under: {BASE_DIR}")


if __name__ == "__main__":
    main()
