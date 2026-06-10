from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on the prepared VisDrone dataset.")
    parser.add_argument("--data", type=Path, default=Path("configs/visdrone_yolo.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("yolo11s.pt"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use '0' for first GPU, 'cpu' for CPU, or leave Ultralytics syntax.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="yolo11s_visdrone")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    weights = resolve_path(args.weights)
    data = resolve_path(args.data)
    project = resolve_path(args.project)

    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
