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

**Goal:** Inspect `EDA Data` for reusable evidence.
**Summary:** The EDA folder contains dataset statistics, full-image YOLO baseline, SAHI baseline, ROI oracle/predata, and failure analysis. Key numbers: 62.36% of labels are small; full-image YOLO small recall is 0.0446; SAHI small recall is 0.1583; ROI analysis says horizon-oriented ROI is often most efficient.
**Files:** `EDA Data/adaptive_rl_sahi/outputs/final/data_research_report.md`, `baseline_metrics.json`, `sahi_metrics.json`, `sahi_test_predata_summary.json`, `test_test_analysis_report.md`.
**Commands:** Read main EDA files with `Get-Content`.
**Next:** Reference these findings when explaining why adaptive slicing is needed.

## 2026-06-10 - Codex

**Goal:** Create tuned same-model config.
**Summary:** Added `configs/default_visdrone_tuned.yaml` and supporting path/inference/RL config files. These keep the same YOLO VisDrone weight but tune output confidence, NMS merge threshold, slice attempts, ROI scale, max slices, and hard-region reward shaping. Smoke-tested inference with the old checkpoint and tuned inference settings; one val image produced 75 boxes and 2 slices.
**Files:** `configs/paths_visdrone_tuned.yaml`, `configs/inference_visdrone_tuned.yaml`, `configs/rl_visdrone_tuned.yaml`, `configs/default_visdrone_tuned.yaml`.
**Commands:** `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --checkpoint runs\dqn_visdrone\best.pt --split val --limit 1` -> passed.
**Next:** Train the tuned DQN into `runs/dqn_visdrone_tuned`.

## 2026-06-10 - Codex

**Goal:** Make slicing more important/effective.
**Summary:** Updated inference behavior so old-overlap ROIs are rejected without ending the entire image. Added density fallback slices based on small and low-confidence detection clusters. Test image `0000006_00611_d_0000002.jpg` now uses 10 slices instead of 3 and produces 158 detections instead of 87.
**Files:** `src/rl_sahi/inference/pipeline.py`.
**Commands:** `python -m py_compile src\rl_sahi\inference\pipeline.py` -> passed; tuned single-image inference -> passed.
**Next:** Inspect output for noisy detections and tune `output_conf`/`min_slice_detections` if needed.

## 2026-06-10 - Codex

**Goal:** Balance slice coverage and visual readability.
**Summary:** Added inference `max_slices` override and tuned the config to cap slices at 5, require at least 3 detections per accepted slice, and raise output confidence to 0.18. Test image now outputs 120 boxes with 5 slices.
**Files:** `configs/inference_visdrone_tuned.yaml`, `src/rl_sahi/inference/config.py`, `src/rl_sahi/inference/pipeline.py`, `scripts/infer.py`.
**Commands:** Compile passed; tuned single-image inference passed.
**Next:** Use `infer.max_slices` as the main knob for visual clutter.

## 2026-06-10 - Codex

**Goal:** Create class-count text report.
**Summary:** Added `scripts/summarize_predictions.py` and generated `runs/infer_visdrone_tuned/class_summary.txt`, summarizing predicted class counts per image and split by full-image vs slice detections.
**Files:** `scripts/summarize_predictions.py`, `runs/infer_visdrone_tuned/class_summary.txt`.
**Commands:** Summary script compile and run passed.
**Next:** Use this report for quick output inspection and thesis tables.

## 2026-06-10 - Codex

**Goal:** Add per-image class-summary files.
**Summary:** Extended `scripts/summarize_predictions.py` with `--per-image-dir`; generated one `.txt` per inferred image under `runs/infer_visdrone_tuned/class_summaries`.
**Files:** `scripts/summarize_predictions.py`, `runs/infer_visdrone_tuned/class_summaries`.
**Commands:** Summary script run wrote 11 per-image files.
**Next:** Re-run after generating more detections.

## 2026-06-10 - Codex

**Goal:** Reduce visualization clutter.
**Summary:** Detection labels/confidence text are no longer drawn. Each VisDrone class now has a fixed box color in `visualize.py`; regenerated the current test image.
**Files:** `src/rl_sahi/inference/visualize.py`.
**Commands:** Compile and tuned single-image inference passed.
**Next:** Keep class details in `.txt` summaries rather than on the image.

## 2026-06-10 - Codex

**Goal:** Add class-color legend.
**Summary:** The summary script now generates `runs/infer_visdrone_tuned/class_color_legend.txt`, mapping class IDs/names to color hints and RGB hex values.
**Files:** `scripts/summarize_predictions.py`, `runs/infer_visdrone_tuned/class_color_legend.txt`.
**Commands:** Summary script rerun passed.
**Next:** Use legend to interpret text-free visual outputs.

## 2026-06-10 - Codex

**Goal:** Restore class-name labels.
**Summary:** Visualization now draws class names only on each box, with no confidence values. Summary files remain available per image.
**Files:** `src/rl_sahi/inference/visualize.py`.
**Commands:** Compile and tuned single-image inference passed.
**Next:** Keep confidence out of the visualization unless explicitly needed.

## 2026-06-10 - Codex

**Goal:** Clean duplicate/wrong-class detections.
**Summary:** Added post-processing refinement: class-wise thresholds, class-aware NMS, then source-aware class-agnostic NMS. Test image final detections dropped from 120 to 85 boxes with same 5 slices.
**Files:** `configs/inference_visdrone_tuned.yaml`, `src/rl_sahi/inference/config.py`, `src/rl_sahi/inference/pipeline.py`, `scripts/infer.py`.
**Commands:** Compile, tuned single-image inference, and summary refresh passed.
**Next:** Tune `agnostic_nms_iou` and class thresholds based on visual quality.

## 2026-06-10 - Codex

**Goal:** Run 20-image tuned smoke test.
**Summary:** Tuned validation inference completed for 20 images and class summaries were refreshed. The tuned output folder now contains 21 summarized images including one earlier test image.
**Files:** `runs/infer_visdrone_tuned/`.
**Commands:** `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --split val --limit 20` -> passed.
**Next:** Review visualization quality across the 20 images.

## 2026-06-10 - Codex

**Goal:** Prepare the repo for GitHub upload.
**Summary:** Expanded `.gitignore` for image/media files, raw datasets, generated caches, outputs, runs, and model artifacts. Added a Vietnamese root `README.md` with setup, GPU verification, data preparation, VisDrone workflow, default workflow notes, and Git upload guidance. Verified representative raw image, generated visualization, and cache paths are ignored. Noted that `git rev-parse --show-toplevel` reports `D:/`, so staging should be done carefully.
**Files:** `.gitignore`, `README.md`, `AGENT_LOG.md`, `NEXT_STEPS.md`, `AGENT_HANDOFF.md`, `.agent-logs/codex.md`.
**Commands:** `git check-ignore -v ...` for sample image/cache/output paths -> ignored; `git rev-parse --show-toplevel` -> `D:/`.
**Next:** Stage only intended project files before pushing to GitHub.
