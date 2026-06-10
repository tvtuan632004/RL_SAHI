from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import BoxRecord, ensure_dirs, load_json, read_all_yolo_labels


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "final" / "figures" / "data"


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def load_index_rows() -> list[dict[str, object]]:
    path = OUTPUT_DIR / "detailed" / "per_image_data_index.json"
    if not path.exists():
        path = OUTPUT_DIR / "per_image_data_index.json"
    if not path.exists():
        raise FileNotFoundError("Missing per-image index. Run run_data_report.py first.")
    return load_json(path)


def select_sample_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def best(predicate, key):
        candidates = [row for row in rows if predicate(row)]
        return max(candidates, key=key) if candidates else None

    choices = [
        best(lambda row: row["split"] == "test", lambda row: row["total_labels"]),
        best(lambda row: row["split"] == "test", lambda row: row["small_labels"]),
        best(lambda row: row["split"] == "test", lambda row: row["border_object_count"]),
        best(lambda row: row["split"] == "test", lambda row: row["small_ratio"]),
    ]
    unique = []
    seen = set()
    for row in choices:
        if row and row["image_id"] not in seen:
            unique.append(row)
            seen.add(row["image_id"])
    if len(unique) < 4:
        # Fallback: fill remaining slots with dense test images not already selected.
        test_rows = [row for row in rows if row["split"] == "test" and row["image_id"] not in seen]
        for row in sorted(test_rows, key=lambda item: item["total_labels"], reverse=True):
            unique.append(row)
            seen.add(row["image_id"])
            if len(unique) == 4:
                break
    return unique[:4]


def records_by_image(records: list[BoxRecord]) -> dict[str, list[BoxRecord]]:
    grouped: dict[str, list[BoxRecord]] = defaultdict(list)
    for record in records:
        grouped[record.image_id].append(record)
    return grouped


def plot_dataset_samples(rows: list[dict[str, object]], grouped_records: dict[str, list[BoxRecord]]) -> None:
    sample_rows = select_sample_rows(rows)
    if not sample_rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()
    colors = {"small": "#ef4444", "medium": "#f59e0b", "large": "#22c55e"}
    labels_used = set()

    for ax, row in zip(axes, sample_rows):
        image = Image.open(row["image_path"]).convert("RGB")
        ax.imshow(image)
        records = grouped_records.get(row["image_id"], [])
        # Draw the smallest objects first and cap the count so dense scenes stay readable.
        drawable = sorted(records, key=lambda record: record.area)[:140]
        for record in drawable:
            color = colors[record.size_bucket]
            rect = patches.Rectangle(
                (record.x1, record.y1),
                record.width,
                record.height,
                linewidth=0.7,
                edgecolor=color,
                facecolor="none",
                alpha=0.9,
                label=record.size_bucket if record.size_bucket not in labels_used else None,
            )
            labels_used.add(record.size_bucket)
            ax.add_patch(rect)
        ax.set_title(f"{row['image_id']}", fontsize=10)
        ax.axis("off")

    # Hide any unused subplot to avoid empty axes/ticks in thesis figures.
    for ax in axes[len(sample_rows):]:
        ax.axis("off")

    handles = [
        patches.Patch(edgecolor=color, facecolor="none", label=name)
        for name, color in colors.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Representative VisDrone Samples with Ground-Truth Boxes")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIGURE_DIR / "dataset_samples_with_boxes.png")
    plt.close(fig)


def plot_split_summary(rows: list[dict[str, object]]) -> None:
    splits = ["train", "val", "test"]
    images = [sum(1 for row in rows if row["split"] == split) for split in splits]
    labels = [sum(row["total_labels"] for row in rows if row["split"] == split) for split in splits]
    small = [sum(row["small_labels"] for row in rows if row["split"] == split) for split in splits]

    x = np.arange(len(splits))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(splits, images, color="#3b82f6")
    axes[0].set_title("Images per Split")
    axes[0].set_ylabel("Image count")

    width = 0.35
    axes[1].bar(x - width / 2, labels, width, label="All labels", color="#64748b")
    axes[1].bar(x + width / 2, small, width, label="Small labels", color="#ef4444")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits)
    axes[1].set_title("Labels per Split")
    axes[1].set_ylabel("Label count")
    axes[1].legend()

    fig.suptitle("VisDrone Dataset Split Summary")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "dataset_split_summary.png")
    plt.close(fig)


def plot_class_distribution(records: list[BoxRecord]) -> None:
    counts = Counter(record.class_name for record in records)
    names = [name for name, _ in counts.most_common()]
    values = [counts[name] for name in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color="#0ea5e9")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Object count")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "class_distribution.png")
    plt.close(fig)


def plot_bbox_distribution(records: list[BoxRecord]) -> None:
    widths = [record.width for record in records]
    heights = [record.height for record in records]
    areas = [record.area for record in records]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(widths, bins=80, color="#3b82f6", alpha=0.86)
    axes[0].set_title("BBox Width")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Object count")
    axes[1].hist(heights, bins=80, color="#10b981", alpha=0.86)
    axes[1].set_title("BBox Height")
    axes[1].set_xlabel("Height (px)")
    axes[2].hist(areas, bins=80, color="#f59e0b", alpha=0.86)
    axes[2].set_title("BBox Area")
    axes[2].set_xlabel("Area (px^2)")
    fig.suptitle("Bounding Box Size Distribution")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bbox_distribution_full.png")
    plt.close(fig)


def plot_object_size_ratio(records: list[BoxRecord]) -> None:
    counts = Counter(record.size_bucket for record in records)
    labels = ["small", "medium", "large"]
    values = [counts[label] for label in labels]
    total = sum(values)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if total > 0:
        axes[0].pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#ef4444", "#f59e0b", "#22c55e"],
        )
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
    axes[0].set_title("Object Size Ratio")
    axes[1].axis("off")
    if total > 0:
        table_data = [[label, values[idx], f"{values[idx] / total * 100:.2f}%"] for idx, label in enumerate(labels)]
    else:
        table_data = [[label, 0, "0.00%"] for label in labels]
    table = axes[1].table(cellText=table_data, colLabels=["Size", "Objects", "Ratio"], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    axes[1].set_title("COCO-Style Size Statistics")
    fig.suptitle("Small / Medium / Large Object Distribution")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "object_size_ratio_full.png")
    plt.close(fig)


def plot_density(rows: list[dict[str, object]]) -> None:
    total = [row["total_labels"] for row in rows]
    small = [row["small_labels"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(total, bins=70, color="#6366f1", alpha=0.86)
    axes[0].set_title("Objects per Image")
    axes[0].set_xlabel("Object count")
    axes[0].set_ylabel("Image count")
    axes[1].hist(small, bins=70, color="#ec4899", alpha=0.86)
    axes[1].set_title("Small Objects per Image")
    axes[1].set_xlabel("Small object count")
    fig.suptitle("Object Density Analysis")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "density_analysis_full.png")
    plt.close(fig)


def plot_spatial_heatmap(records: list[BoxRecord]) -> None:
    small = [record for record in records if record.size_bucket == "small"]
    panels = [
        ("All Object Centers", records, "viridis"),
        ("Small Object Centers", small, "magma"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (title, subset, cmap) in zip(axes, panels):
        x = [record.norm_center_x for record in subset]
        y = [record.norm_center_y for record in subset]
        heat = ax.hist2d(x, y, bins=70, range=[[0, 1], [0, 1]], cmap=cmap)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        fig.colorbar(heat[3], ax=ax, label="Object count")
    fig.suptitle("Spatial Distribution Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "spatial_heatmap_full.png")
    plt.close(fig)


def plot_border_analysis(records: list[BoxRecord], threshold_px: int) -> None:
    distances = [
        min(record.x1, record.y1, record.image_width - record.x2, record.image_height - record.y2)
        for record in records
    ]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(distances, bins=80, color="#14b8a6", alpha=0.86)
    ax.axvline(threshold_px, color="#ef4444", linestyle="--", label=f"{threshold_px}px border threshold")
    ax.set_title("Distance-to-Border Distribution")
    ax.set_xlabel("Minimum distance to image border (px)")
    ax.set_ylabel("Object count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "border_analysis_full.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready data-only figures.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to include.")
    parser.add_argument("--border-threshold-px", type=int, default=32)
    args = parser.parse_args()

    ensure_dirs(FIGURE_DIR)
    setup_plot_style()
    rows = load_index_rows()
    rows = [row for row in rows if row["split"] in set(args.splits)]
    records = read_all_yolo_labels(args.splits)
    grouped = records_by_image(records)

    plot_dataset_samples(rows, grouped)
    plot_split_summary(rows)
    plot_class_distribution(records)
    plot_bbox_distribution(records)
    plot_object_size_ratio(records)
    plot_density(rows)
    plot_spatial_heatmap(records)
    plot_border_analysis(records, args.border_threshold_px)

    print(f"Data figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
