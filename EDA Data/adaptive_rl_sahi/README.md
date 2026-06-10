# Data Analysis and YOLO Baseline for Adaptive ROI-SAHI

This folder is independent from the older project code. It reads the existing VisDrone data from:

- `../../data/raw`
- `../../data/processed`

It does not modify the old `src`, `scripts`, `tests`, or config files.

## Phase Coverage

- Dataset validation and statistics
- Object size, spatial, density, and border EDA
- YOLO validation baseline
- Small-object failure and confidence analysis
- Heuristic ROI coverage simulation

Reinforcement learning is intentionally excluded from this phase.

## Run

From this folder:

```bash
python run_eda.py
python run_eda.py --splits val
python run_data_report.py --splits train val test
python run_baseline.py --model yolo11s.pt --split val --sample-limit 50
python run_baseline.py --model yolo11s.pt --split test --class-agnostic
python run_test_analysis.py --split test
python run_prepare_sahi_predata.py --split test
python run_sahi_baseline.py --model yolo11s.pt --split test --device cpu
python run_roi_analysis.py
```

`run_eda.py` uses `train val test` by default. Use `--splits val` for a quick smoke test.
For full validation baseline, omit `--sample-limit`.

## Baseline Model Choice

The baseline intentionally uses pretrained `yolo11s.pt` without VisDrone fine-tuning. This keeps full-image YOLO imperfect enough to study:

- small-object failures
- confidence behavior
- whether SAHI and adaptive ROI slicing are justified

Because pretrained YOLO uses COCO class IDs, strict VisDrone class-ID matching is not a perfect dataset-specific mAP. Treat this baseline primarily as failure-analysis evidence, not as a final VisDrone leaderboard score.

```bash
python run_baseline.py --model yolo11s.pt --split val
python run_baseline.py --model yolo11s.pt --split val --class-agnostic
```

`run_train_yolo.py` exists only as an optional future utility and is not part of the current ROI-SAHI baseline workflow.

## Outputs

- `outputs/final/`: core report files and final figures.
- `outputs/detailed/`: false positives, false negatives, latency, and detailed per-image metadata.
- `outputs/heavy/`: large prediction/pre-data/oracle files.
- `outputs/archive/`: older validation-only and auxiliary EDA outputs.
