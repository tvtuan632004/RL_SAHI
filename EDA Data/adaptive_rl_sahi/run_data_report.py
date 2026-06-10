from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from utils import (
    BoxRecord,
    box_center_inside_roi,
    ensure_dirs,
    read_yolo_labels,
    roi_area_ratio,
    roi_boxes,
    save_json,
    split_paths,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
ROI_STRATEGIES = ("horizon", "center", "uncertain", "density")


def records_by_image(split: str) -> tuple[dict[str, Path], dict[str, list[BoxRecord]]]:
    _, _, image_dir, _ = split_paths(split)
    image_paths = {path.stem: path for path in sorted(image_dir.glob("*.jpg"))}
    grouped: dict[str, list[BoxRecord]] = defaultdict(list)
    for record in read_yolo_labels(split):
        grouped[record.image_id].append(record)
    return image_paths, grouped


def density_level(total_labels: int) -> str:
    if total_labels >= 100:
        return "high"
    if total_labels >= 30:
        return "medium"
    return "low"


def small_level(small_labels: int, total_labels: int) -> str:
    ratio = small_labels / total_labels if total_labels else 0.0
    if small_labels >= 50 or ratio >= 0.75:
        return "many_small"
    if small_labels >= 10 or ratio >= 0.4:
        return "some_small"
    return "few_small"


def border_count(records: list[BoxRecord], threshold_px: int) -> int:
    count = 0
    for record in records:
        distance = min(record.x1, record.y1, record.image_width - record.x2, record.image_height - record.y2)
        if distance <= threshold_px:
            count += 1
    return count


def dominant_classes(records: list[BoxRecord], top_k: int = 3) -> list[str]:
    counts = Counter(record.class_name for record in records)
    return [name for name, _ in counts.most_common(top_k)]


def roi_oracle_for_image(records: list[BoxRecord]) -> dict[str, object]:
    if not records:
        return {"strategies": [], "best_by_small_efficiency": None, "best_by_small_coverage": None}

    width = records[0].image_width
    height = records[0].image_height
    small_objects = [record for record in records if record.size_bucket == "small"]
    strategies = []
    for strategy in ROI_STRATEGIES:
        rois = roi_boxes(strategy, width, height)
        covered = sum(1 for record in small_objects if any(box_center_inside_roi(record, roi) for roi in rois))
        area_ratio = roi_area_ratio(rois, width, height)
        coverage = covered / len(small_objects) if small_objects else 0.0
        strategies.append(
            {
                "strategy": strategy,
                "boxes_xyxy": [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in rois],
                "small_objects": len(small_objects),
                "small_objects_covered": covered,
                "small_object_coverage": coverage,
                "roi_area_ratio": area_ratio,
                "efficiency_score": coverage / area_ratio if area_ratio else 0.0,
            }
        )

    best_efficiency = max(strategies, key=lambda item: item["efficiency_score"])["strategy"] if strategies else None
    best_coverage = max(strategies, key=lambda item: item["small_object_coverage"])["strategy"] if strategies else None
    return {
        "strategies": strategies,
        "best_by_small_efficiency": best_efficiency,
        "best_by_small_coverage": best_coverage,
    }


def build_index_for_split(split: str, border_threshold_px: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    image_paths, grouped = records_by_image(split)
    index_rows = []
    roi_rows = []
    for image_id, image_path in sorted(image_paths.items()):
        records = grouped.get(image_id, [])
        width = records[0].image_width if records else 0
        height = records[0].image_height if records else 0
        by_size = Counter(record.size_bucket for record in records)
        by_class = Counter(record.class_name for record in records)
        total = len(records)
        small = by_size.get("small", 0)
        border_objects = border_count(records, border_threshold_px)
        roi_oracle = roi_oracle_for_image(records)

        index_rows.append(
            {
                "image_id": image_id,
                "split": split,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "total_labels": total,
                "small_labels": small,
                "medium_labels": by_size.get("medium", 0),
                "large_labels": by_size.get("large", 0),
                "small_ratio": small / total if total else 0.0,
                "border_object_count": border_objects,
                "border_object_ratio": border_objects / total if total else 0.0,
                "density_level": density_level(total),
                "small_object_level": small_level(small, total),
                "dominant_classes": dominant_classes(records),
                "class_distribution": dict(sorted(by_class.items())),
                "roi_best_by_small_efficiency": roi_oracle["best_by_small_efficiency"],
                "roi_best_by_small_coverage": roi_oracle["best_by_small_coverage"],
            }
        )
        roi_rows.append({"image_id": image_id, "split": split, **roi_oracle})
    return index_rows, roi_rows


def write_index_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "split",
        "image_path",
        "width",
        "height",
        "total_labels",
        "small_labels",
        "medium_labels",
        "large_labels",
        "small_ratio",
        "border_object_count",
        "border_object_ratio",
        "density_level",
        "small_object_level",
        "dominant_classes",
        "roi_best_by_small_efficiency",
        "roi_best_by_small_coverage",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in fieldnames}
            csv_row["dominant_classes"] = "|".join(row["dominant_classes"])
            writer.writerow(csv_row)


def build_difficulty_groups(rows: list[dict[str, object]], top_k: int) -> dict[str, list[dict[str, object]]]:
    def compact(row: dict[str, object]) -> dict[str, object]:
        return {
            "image_id": row["image_id"],
            "split": row["split"],
            "image_path": row["image_path"],
            "total_labels": row["total_labels"],
            "small_labels": row["small_labels"],
            "small_ratio": row["small_ratio"],
            "border_object_count": row["border_object_count"],
            "border_object_ratio": row["border_object_ratio"],
            "density_level": row["density_level"],
            "small_object_level": row["small_object_level"],
            "dominant_classes": row["dominant_classes"],
        }

    return {
        "high_density": [compact(row) for row in sorted(rows, key=lambda r: r["total_labels"], reverse=True)[:top_k]],
        "many_small_objects": [compact(row) for row in sorted(rows, key=lambda r: r["small_labels"], reverse=True)[:top_k]],
        "high_small_ratio": [compact(row) for row in sorted(rows, key=lambda r: r["small_ratio"], reverse=True)[:top_k]],
        "border_heavy": [compact(row) for row in sorted(rows, key=lambda r: r["border_object_count"], reverse=True)[:top_k]],
        "low_density": [compact(row) for row in sorted(rows, key=lambda r: r["total_labels"])[:top_k]],
    }


def aggregate_report(rows: list[dict[str, object]], roi_rows: list[dict[str, object]], splits: list[str]) -> dict[str, object]:
    split_summary = {}
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        split_summary[split] = {
            "images": len(split_rows),
            "total_labels": sum(row["total_labels"] for row in split_rows),
            "small_labels": sum(row["small_labels"] for row in split_rows),
            "medium_labels": sum(row["medium_labels"] for row in split_rows),
            "large_labels": sum(row["large_labels"] for row in split_rows),
            "avg_labels_per_image": (
                sum(row["total_labels"] for row in split_rows) / len(split_rows) if split_rows else 0.0
            ),
            "avg_small_labels_per_image": (
                sum(row["small_labels"] for row in split_rows) / len(split_rows) if split_rows else 0.0
            ),
        }

    density_counts = Counter(row["density_level"] for row in rows)
    small_level_counts = Counter(row["small_object_level"] for row in rows)
    roi_eff_wins = Counter(row["best_by_small_efficiency"] for row in roi_rows if row["best_by_small_efficiency"])
    roi_cov_wins = Counter(row["best_by_small_coverage"] for row in roi_rows if row["best_by_small_coverage"])

    total_labels = sum(row["total_labels"] for row in rows)
    total_small = sum(row["small_labels"] for row in rows)
    return {
        "splits": splits,
        "total_images": len(rows),
        "total_labels": total_labels,
        "total_small_labels": total_small,
        "small_label_ratio": total_small / total_labels if total_labels else 0.0,
        "split_summary": split_summary,
        "density_level_counts": dict(sorted(density_counts.items())),
        "small_object_level_counts": dict(sorted(small_level_counts.items())),
        "roi_best_by_small_efficiency_counts": dict(sorted(roi_eff_wins.items())),
        "roi_best_by_small_coverage_counts": dict(sorted(roi_cov_wins.items())),
    }


def write_markdown_report(report: dict[str, object], difficulty_groups: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# Data Research Report",
        "",
        f"- Splits: {', '.join(report['splits'])}",
        f"- Total images: {report['total_images']}",
        f"- Total labels: {report['total_labels']}",
        f"- Total small labels: {report['total_small_labels']}",
        f"- Small label ratio: {report['small_label_ratio']:.4f}",
        "",
        "## Split Summary",
    ]
    for split, values in report["split_summary"].items():
        lines.append(
            f"- {split}: images={values['images']}, labels={values['total_labels']}, "
            f"small={values['small_labels']}, avg_labels/image={values['avg_labels_per_image']:.2f}"
        )
    lines.extend(["", "## Difficulty Groups"])
    for group_name, group_rows in difficulty_groups.items():
        if not group_rows:
            continue
        first = group_rows[0]
        lines.append(
            f"- {group_name}: top image={first['image_id']} "
            f"({first['total_labels']} labels, {first['small_labels']} small)"
        )
    lines.extend(
        [
            "",
            "## ROI Oracle",
            f"- Best-by-efficiency counts: {report['roi_best_by_small_efficiency_counts']}",
            f"- Best-by-coverage counts: {report['roi_best_by_small_coverage_counts']}",
            "",
            "This report is data-only and does not use model predictions.",
        ]
    )
    (OUTPUT_DIR / "data_research_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data-only per-image index, difficulty groups, and ROI oracle.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Splits to include.")
    parser.add_argument("--border-threshold-px", type=int, default=32, help="Distance threshold for border objects.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K images per difficulty group.")
    args = parser.parse_args()

    ensure_dirs(OUTPUT_DIR)
    all_rows: list[dict[str, object]] = []
    all_roi_rows: list[dict[str, object]] = []
    for split in args.splits:
        print(f"Building data index for split={split}")
        rows, roi_rows = build_index_for_split(split, args.border_threshold_px)
        all_rows.extend(rows)
        all_roi_rows.extend(roi_rows)

    difficulty_groups = build_difficulty_groups(all_rows, args.top_k)
    report = aggregate_report(all_rows, all_roi_rows, args.splits)

    save_json(all_rows, OUTPUT_DIR / "per_image_data_index.json")
    write_index_csv(all_rows, OUTPUT_DIR / "per_image_data_index.csv")
    save_json(difficulty_groups, OUTPUT_DIR / "difficulty_groups.json")
    save_json(all_roi_rows, OUTPUT_DIR / "roi_oracle_per_image.json")
    save_json(report, OUTPUT_DIR / "data_research_report.json")
    write_markdown_report(report, difficulty_groups)

    print(f"Data report complete. Images={report['total_images']}, labels={report['total_labels']}")


if __name__ == "__main__":
    main()

