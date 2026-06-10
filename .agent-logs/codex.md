# Codex Log

## 2026-06-09

- Reviewed the RL-SAHI project structure and explained the pipeline.
- Helped test inference, validation cache generation, hard-region cache generation, and short DQN training.
- Drafted thesis text for RL input data and data summary sections.
- Proposed a 5-6 slide presentation structure.
- Created repo-level cross-agent handoff files.
- Added `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` so fresh agents can discover the handoff workflow.
- Inspected `archive/` and confirmed it contains VisDrone train/val/test-dev/test-challenge original-format data.
- Confirmed 30,000-episode DQN training completed and checkpoints were updated on 2026-06-10.

Next recommended action: run validation inference with the newly trained checkpoint, then add quantitative evaluation.

## 2026-06-10

- Checked whether conda env `tvtuan` and RL-SAHI are using GPU/CUDA.
- Confirmed `nvidia-smi` sees an NVIDIA GeForce RTX 5060 Laptop GPU.
- Confirmed `conda run -n tvtuan` uses `torch 2.8.0+cpu`; `torch.cuda.is_available()` is `False`, `torch.version.cuda` is `None`, and device count is 0.
- Inspected config/code device handling: blank `device: ""` values defer to auto-detection, and training/inference choose CUDA only when PyTorch reports it available.
- Noted an unrelated sandbox permission error when importing `ultralytics` because it tried to read `%AppData%\Ultralytics\settings.json`.

- Installed CUDA-enabled PyTorch in `tvtuan` using the PyTorch CUDA 12.8 wheel index.
- Verified `torch 2.11.0+cu128`, CUDA available, CUDA runtime `12.8`, one GPU, `NVIDIA GeForce RTX 5060 Laptop GPU`, and test tensor allocation on `cuda:0`.
- Verified `from ultralytics import YOLO` succeeds when run with normal permissions.
- Ran `pip check`; noted dependency conflicts for unrelated packages such as `unsloth`/`unsloth-zoo` and `whisperx` that expect older torch versions.
- Fixed `FeatureCollector` so first-predict warmup does not duplicate YOLO feature vectors.
- Rebuilt the first val detection cache and confirmed `python scripts/infer.py --split val --limit 1` passes with 21 boxes and 3 slices.
- Added class-name and confidence labels to inference visualizations; verified on one validation image.

Next recommended action: run full validation inference with `python scripts/infer.py --split val` to regenerate all visualizations with labels; use a separate environment for non-RL-SAHI workloads that require older torch.
## 2026-06-10 - Codex

**Goal:** Fix wrong detection labels caused by using a COCO YOLO weight.
**Summary:** Verified `configs/paths.yaml` points to `yolo11s.pt` and the attached inference image shows COCO classes, not VisDrone classes. Added a VisDrone YOLO data config, a separate `default_visdrone.yaml` RL-SAHI config, and `scripts/train_yolo.py`. Confirmed the existing VisDrone YOLO weight reports the correct 10 class names and can generate a validation detection cache.
**Files:** `configs/visdrone_yolo.yaml`, `configs/paths_visdrone.yaml`, `configs/default_visdrone.yaml`, `scripts/train_yolo.py`, `data/cache_visdrone/detections/val/0000001_02999_d_0000005.npz`.
**Commands:** `python -m py_compile scripts/train_yolo.py` -> passed; `python scripts/detect.py --config configs/default_visdrone.yaml --split val --limit 1 --overwrite` -> passed.
**Next:** Rebuild all VisDrone caches and train a new DQN with `configs/default_visdrone.yaml`.

## 2026-06-10 - Codex

**Goal:** Record completed VisDrone DQN training.
**Summary:** User reported completion of `ep=30000/30000` for the VisDrone DQN run. Best checkpoint is now `runs/dqn_visdrone/best.pt`.
**Files:** `runs/dqn_visdrone/best.pt`, `AGENT_LOG.md`, `NEXT_STEPS.md`, `AGENT_HANDOFF.md`, `.agent-logs/codex.md`.
**Commands:** User-reported `python scripts/train.py --config configs/default_visdrone.yaml --split train --episodes 30000` -> completed.
**Next:** Run limited validation inference with `configs/default_visdrone.yaml`.

## 2026-06-10 - Codex

**Goal:** Prepare the repo for GitHub upload.
**Summary:** Expanded `.gitignore` for image/media files, raw datasets, generated caches, outputs, runs, and model artifacts. Added a Vietnamese root `README.md` with setup, GPU verification, data preparation, VisDrone workflow, default workflow notes, and Git upload guidance. Verified representative raw image, generated visualization, and cache paths are ignored. Noted that `git rev-parse --show-toplevel` reports `D:/`, so staging should be done carefully.
**Files:** `.gitignore`, `README.md`, `AGENT_LOG.md`, `NEXT_STEPS.md`, `AGENT_HANDOFF.md`, `.agent-logs/codex.md`.
**Commands:** `git check-ignore -v ...` for sample image/cache/output paths -> ignored; `git rev-parse --show-toplevel` -> `D:/`.
**Next:** Stage only intended project files before pushing to GitHub.
