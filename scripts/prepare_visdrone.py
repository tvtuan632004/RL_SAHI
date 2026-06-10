from __future__ import annotations

"""Download/organize the VisDrone dataset into the layout this project expects.

Target layout (from configs/paths.yaml):
    data/raw/images/<split>/*.jpg
    data/raw/labels/<split>/*.txt   # YOLO: class cx cy w h (normalized), class = category-1

Handles two common source layouts automatically:
  * Original VisDrone:  VisDrone2019-DET-<split>/{images,annotations}
                        annotations: x,y,w,h,score,category,trunc,occ  (comma separated)
  * YOLO mirror:        images/<split>/*.jpg  +  labels/<split>/*.txt  (already normalized)

VisDrone -> YOLO conversion rule (same as Ultralytics, matches existing test labels):
  skip rows whose score field == 0 (ignored regions); class = category - 1.

Usage:
    # download via kagglehub, then organize every split it finds
    python scripts/prepare_visdrone.py

    # use an already-downloaded folder instead of downloading
    python scripts/prepare_visdrone.py --source "C:/path/to/visdrone-dataset"

    # only some splits, limit count (quick test)
    python scripts/prepare_visdrone.py --splits train val --limit 50
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_split(path_parts: list[str]) -> str | None:
    joined = "/".join(p.lower() for p in path_parts)
    if "test-challenge" in joined:
        return None  # no labels, skip
    if "train" in joined:
        return "train"
    if "val" in joined:
        return "val"
    if "test" in joined:  # covers test-dev
        return "test"
    return None


def find_label_file(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for src_key in ("images", "Images"):
        for dst_key in ("annotations", "labels", "Annotations", "Labels"):
            if src_key in parts:
                new_parts = [dst_key if p == src_key else p for p in parts]
                candidate = Path(*new_parts).with_suffix(".txt")
                if candidate.exists():
                    return candidate
    sibling = image_path.with_suffix(".txt")
    return sibling if sibling.exists() else None


def convert_label(label_path: Path, image_path: Path) -> list[str]:
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    is_visdrone = "," in lines[0]
    if not is_visdrone:
        # already YOLO: class cx cy w h  -> keep first 5 tokens
        out = []
        for ln in lines:
            toks = ln.split()
            if len(toks) >= 5:
                out.append(" ".join(toks[:5]) + "\n")
        return out

    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    dw, dh = 1.0 / w, 1.0 / h
    out = []
    for ln in lines:
        row = ln.split(",")
        if len(row) < 6:
            continue
        if row[4].strip() == "0":  # ignored region
            continue
        try:
            x, y, bw, bh = (int(float(v)) for v in row[:4])
            cls = int(float(row[5])) - 1
        except ValueError:
            continue
        if cls < 0 or bw <= 0 or bh <= 0:
            continue
        cx, cy = (x + bw / 2) * dw, (y + bh / 2) * dh
        out.append(f"{cls} {cx:.6f} {cy:.6f} {bw * dw:.6f} {bh * dh:.6f}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VisDrone into the project's data/raw layout.")
    parser.add_argument("--source", type=Path, default=None, help="Path to an already-downloaded dataset root.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=None, choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Max images per split (quick test).")
    parser.add_argument("--overwrite", action="store_true", help="Re-copy/convert even if target exists.")
    args = parser.parse_args()

    if args.source is not None:
        source = args.source
    else:
        import kagglehub

        source = Path(kagglehub.dataset_download("kushagrapandya/visdrone-dataset"))
    print(f"[prepare] source: {source}")

    cfg = load_default_config(args.config, ROOT)
    image_root = cfg.path_value("image_root")
    label_root = cfg.path_value("label_root")

    all_images = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"[prepare] found {len(all_images)} images under source")

    wanted = set(args.splits) if args.splits else {"train", "val", "test"}
    counts: dict[str, int] = {s: 0 for s in wanted}
    skipped_no_label = 0

    for image_path in sorted(all_images):
        split = resolve_split(list(image_path.parts))
        if split is None or split not in wanted:
            continue
        if args.limit is not None and counts[split] >= args.limit:
            continue

        out_img = image_root / split / image_path.name
        out_lbl = label_root / split / (image_path.stem + ".txt")
        if out_img.exists() and out_lbl.exists() and not args.overwrite:
            counts[split] += 1
            continue

        label_path = find_label_file(image_path)
        if label_path is None:
            skipped_no_label += 1
            continue
        lines = convert_label(label_path, image_path)

        out_img.parent.mkdir(parents=True, exist_ok=True)
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_img)
        out_lbl.write_text("".join(lines), encoding="utf-8")
        counts[split] += 1
        total = sum(counts.values())
        if total == 1 or total % 200 == 0:
            print(f"[prepare] {split}: {counts[split]} done (latest -> {out_img})")

    print("[prepare] done.")
    for split in sorted(wanted):
        print(f"  {split}: {counts[split]} images -> {image_root / split}")
    if skipped_no_label:
        print(f"  skipped (no matching label): {skipped_no_label}")


if __name__ == "__main__":
    main()
