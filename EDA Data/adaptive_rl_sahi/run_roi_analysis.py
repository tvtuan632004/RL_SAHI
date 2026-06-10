from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    BoxRecord,
    ensure_dirs,
    load_json,
    read_yolo_labels,
    roi_area_ratio,
    roi_boxes,
    save_json,
    box_center_inside_roi,
)


BASE_DIR = Path(__file__).resolve().parent
ROI_DIR = BASE_DIR / "roi_analysis"
OUTPUT_DIR = BASE_DIR / "outputs"
STRATEGIES = ("horizon", "center", "uncertain", "density")


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


def load_missed_objects() -> list[BoxRecord]:
    path = OUTPUT_DIR / "baseline_false_negatives.json"
    if not path.exists():
        return []
    return [dict_to_box_record(item) for item in load_json(path)]


def coverage_for_records(records: list[BoxRecord]) -> dict[str, dict[str, float]]:
    by_image = defaultdict(list)
    for record in records:
        by_image[record.image_id].append(record)

    results: dict[str, dict[str, float]] = {}
    for strategy in STRATEGIES:
        covered = 0
        total = len(records)
        weighted_area = []
        for image_id, image_records in by_image.items():
            width = image_records[0].image_width
            height = image_records[0].image_height
            rois = roi_boxes(strategy, width, height)
            weighted_area.append(roi_area_ratio(rois, width, height))
            for record in image_records:
                if any(box_center_inside_roi(record, roi) for roi in rois):
                    covered += 1
        coverage_ratio = covered / total if total else 0.0
        area_ratio = float(np.mean(weighted_area)) if weighted_area else 0.0
        results[strategy] = {
            "covered_objects": covered,
            "total_objects": total,
            "coverage_ratio": coverage_ratio,
            "roi_area_ratio": area_ratio,
            "efficiency_score": coverage_ratio / area_ratio if area_ratio else 0.0,
        }
    return results


def plot_coverage(results: dict[str, dict[str, float]], output_path: Path, title: str) -> None:
    strategies = list(results.keys())
    coverage = [results[s]["coverage_ratio"] for s in strategies]
    area = [results[s]["roi_area_ratio"] for s in strategies]
    efficiency = [results[s]["efficiency_score"] for s in strategies]

    x = np.arange(len(strategies))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(x - width, coverage, width, label="Coverage ratio", color="#3b82f6")
    ax1.bar(x, area, width, label="ROI area ratio", color="#f59e0b")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Ratio")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)

    ax2 = ax1.twinx()
    ax2.bar(x + width, efficiency, width, label="Efficiency score", color="#10b981")
    ax2.set_ylabel("Coverage / area")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_roi_insights(small_results: dict, miss_results: dict) -> None:
    best_small = max(small_results.items(), key=lambda item: item[1]["efficiency_score"])[0] if small_results else "n/a"
    best_miss = max(miss_results.items(), key=lambda item: item[1]["efficiency_score"])[0] if miss_results else "n/a"
    text = [
        "# ROI Analysis Insights",
        "",
        f"- Most efficient ROI strategy for all small objects: {best_small}.",
        f"- Most efficient ROI strategy for missed objects: {best_miss}.",
        "- Prefer strategies with high coverage and low area ratio because they reduce redundant slicing cost.",
        "- If missed-object coverage is high for a strategy, adaptive ROI slicing is empirically justified for that region.",
        "",
        "RL is intentionally excluded in this phase.",
    ]
    (OUTPUT_DIR / "roi_research_insights.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    ensure_dirs(ROI_DIR, OUTPUT_DIR)
    setup_plot_style()

    val_records = read_yolo_labels("val")
    small_records = [record for record in val_records if record.size_bucket == "small"]
    missed_records = load_missed_objects()
    missed_small = [record for record in missed_records if record.size_bucket == "small"]

    small_results = coverage_for_records(small_records)
    miss_results = coverage_for_records(missed_small)

    save_json({"small_object_coverage": small_results, "missed_small_object_coverage": miss_results}, OUTPUT_DIR / "roi_metrics.json")
    plot_coverage(small_results, ROI_DIR / "roi_coverage.png", "ROI Coverage for Validation Small Objects")
    plot_coverage(miss_results, ROI_DIR / "miss_coverage.png", "ROI Coverage for YOLO Missed Small Objects")
    write_roi_insights(small_results, miss_results)

    print("ROI analysis complete.")
    if not missed_records:
        print("No baseline_false_negatives.json found yet; miss coverage plot is empty until run_baseline.py is executed.")


if __name__ == "__main__":
    main()

