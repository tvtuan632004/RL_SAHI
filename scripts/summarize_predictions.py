from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.inference.visualize import CLASS_COLORS

VISDRONE_NAMES = {
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


COLOR_HINTS = {
    0: "blue",
    1: "light blue",
    2: "yellow",
    3: "green",
    4: "purple",
    5: "orange",
    6: "cyan",
    7: "pink",
    8: "red",
    9: "brown/orange",
}


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def bgr_to_hex(color: tuple[int, int, int]) -> str:
    b, g, r = color
    return f"#{r:02X}{g:02X}{b:02X}"


def write_class_legend(path: Path) -> None:
    lines = [
        "VisDrone Class Color Legend",
        "",
        "Detection boxes use one fixed color per class.",
        "ROI slice boxes are drawn separately in red/orange.",
        "",
        "class_id\tclass_name\tcolor_hint\thex_rgb",
    ]
    for cls, name in VISDRONE_NAMES.items():
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (180, 180, 180)
        lines.append(f"{cls}\t{name}\t{COLOR_HINTS.get(cls, 'gray')}\t{bgr_to_hex(color)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def read_prediction_file(path: Path) -> tuple[Counter[int], Counter[int], Counter[int]]:
    total: Counter[int] = Counter()
    full: Counter[int] = Counter()
    sliced: Counter[int] = Counter()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        source = int(float(parts[6]))
        total[cls] += 1
        if source == 0:
            full[cls] += 1
        else:
            sliced[cls] += 1
    return total, full, sliced


def format_counter(counter: Counter[int]) -> str:
    if not counter:
        return "none"
    parts = []
    for cls in sorted(counter):
        name = VISDRONE_NAMES.get(cls, f"class_{cls}")
        parts.append(f"{name}={counter[cls]}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize predicted class counts for each inference image.")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("runs/infer_visdrone_tuned/detections"),
        help="Folder containing inference .txt files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/infer_visdrone_tuned/class_summary.txt"),
        help="Output summary .txt file.",
    )
    parser.add_argument(
        "--per-image-dir",
        type=Path,
        default=Path("runs/infer_visdrone_tuned/class_summaries"),
        help="Folder for one summary .txt file per image.",
    )
    parser.add_argument(
        "--legend-out",
        type=Path,
        default=Path("runs/infer_visdrone_tuned/class_color_legend.txt"),
        help="Output class color legend .txt file.",
    )
    args = parser.parse_args()

    pred_dir = resolve_path(args.pred_dir)
    out_path = resolve_path(args.out)
    per_image_dir = resolve_path(args.per_image_dir)
    legend_path = resolve_path(args.legend_out)
    files = sorted(pred_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No prediction .txt files found under {pred_dir}")

    dataset_total: Counter[int] = Counter()
    dataset_full: Counter[int] = Counter()
    dataset_slice: Counter[int] = Counter()
    rows: list[str] = []

    for path in files:
        total, full, sliced = read_prediction_file(path)
        dataset_total.update(total)
        dataset_full.update(full)
        dataset_slice.update(sliced)
        rows.append(f"Image: {path.stem}.jpg")
        rows.append(f"  Total boxes: {sum(total.values())}")
        rows.append(f"  Classes: {format_counter(total)}")
        rows.append(f"  Full-image: {format_counter(full)}")
        rows.append(f"  Slice: {format_counter(sliced)}")
        rows.append("")

        image_lines = [
            f"Image: {path.stem}.jpg",
            f"Prediction file: {path}",
            "",
            f"Total boxes: {sum(total.values())}",
            f"Classes: {format_counter(total)}",
            "",
            "By Source",
            f"  Full-image: {format_counter(full)}",
            f"  Slice: {format_counter(sliced)}",
            "",
            "Class Table",
            "class_id\tclass_name\ttotal\tfull_image\tslice",
        ]
        for cls in sorted(set(total) | set(full) | set(sliced)):
            image_lines.append(
                f"{cls}\t{VISDRONE_NAMES.get(cls, f'class_{cls}')}\t{total[cls]}\t{full[cls]}\t{sliced[cls]}"
            )
        per_image_dir.mkdir(parents=True, exist_ok=True)
        (per_image_dir / f"{path.stem}.txt").write_text("\n".join(image_lines), encoding="utf-8")

    lines = [
        "RL-SAHI Prediction Class Summary",
        f"Prediction folder: {pred_dir}",
        f"Images summarized: {len(files)}",
        "",
        "Overall",
        f"  Total boxes: {sum(dataset_total.values())}",
        f"  Classes: {format_counter(dataset_total)}",
        f"  Full-image: {format_counter(dataset_full)}",
        f"  Slice: {format_counter(dataset_slice)}",
        "",
        "Per Image",
        "",
    ]
    lines.extend(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    write_class_legend(legend_path)
    print(f"[summary] wrote {out_path}")
    print(f"[summary] wrote {len(files)} per-image files under {per_image_dir}")
    print(f"[summary] wrote {legend_path}")


if __name__ == "__main__":
    main()
