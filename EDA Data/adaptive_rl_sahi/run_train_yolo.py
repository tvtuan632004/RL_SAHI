from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from utils import PROCESSED_DATA_ROOT, VISDRONE_CLASS_NAMES, ensure_dirs, save_json


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "datasets"
BASELINE_DIR = BASE_DIR / "baseline"
OUTPUT_DIR = BASE_DIR / "outputs"


def build_visdrone_yaml() -> Path:
    """Create a YOLO dataset YAML whose class order matches converted VisDrone labels."""
    ensure_dirs(DATASET_DIR)
    data_yaml = {
        "path": str(PROCESSED_DATA_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(VISDRONE_CLASS_NAMES),
        "names": [VISDRONE_CLASS_NAMES[idx] for idx in range(len(VISDRONE_CLASS_NAMES))],
    }
    yaml_path = DATASET_DIR / "visdrone_yolo.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    save_json(data_yaml, OUTPUT_DIR / "visdrone_yolo_dataset.json")
    return yaml_path


def train_yolo(args: argparse.Namespace) -> Path:
    ensure_dirs(BASELINE_DIR, OUTPUT_DIR)
    local_ultralytics_dir = OUTPUT_DIR / ".ultralytics"
    local_ultralytics_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_ultralytics_dir))

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics before training YOLO.") from exc

    yaml_path = build_visdrone_yaml()
    model_ref = Path(args.model)
    if model_ref.suffix == ".pt" and len(model_ref.parts) == 1:
        local_model = BASELINE_DIR / args.model
        model_ref = local_model if local_model.exists() else model_ref

    model = YOLO(str(model_ref))
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(BASELINE_DIR / "training_runs"),
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        verbose=True,
    )
    run_dir = Path(results.save_dir)
    best_weight = run_dir / "weights" / "best.pt"
    last_weight = run_dir / "weights" / "last.pt"
    save_json(
        {
            "dataset_yaml": str(yaml_path),
            "run_dir": str(run_dir),
            "best_weight": str(best_weight),
            "last_weight": str(last_weight),
            "class_order": VISDRONE_CLASS_NAMES,
        },
        OUTPUT_DIR / "training_run.json",
    )
    return best_weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on converted VisDrone labels.")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting YOLO checkpoint.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device, 'cpu', or Ultralytics device string.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--name", default="yolo11n_visdrone", help="Training run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting the same run name.")
    parser.add_argument("--only-yaml", action="store_true", help="Only create datasets/visdrone_yolo.yaml.")
    args = parser.parse_args()

    yaml_path = build_visdrone_yaml()
    print(f"Dataset YAML ready: {yaml_path}")
    if args.only_yaml:
        return

    best_weight = train_yolo(args)
    print(f"Training complete. Best weight: {best_weight}")
    print("Evaluate with:")
    print(f'python run_baseline.py --model "{best_weight}" --split val')


if __name__ == "__main__":
    main()

