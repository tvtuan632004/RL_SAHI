# Agent Log

## 2026-06-09 - Codex

**Goal:** Understand and test the RL-SAHI project.
**Summary:** Reviewed project architecture, explained the YOLO + hard-region cache + DQN adaptive slicing pipeline, and confirmed that train cache already exists.
**Files:** Project source and config files inspected; no source code changes.
**Commands:** `python scripts/infer.py --split train --limit 3` -> passed; `python scripts/detect.py --split val --limit 5` -> passed; `python scripts/hard_region.py --split val --limit 5` -> passed; `python scripts/infer.py --split val --limit 5` -> passed; `python scripts/train.py --split train --episodes 5 --limit 20` -> passed.
**Next:** Add quantitative evaluation and prepare final slides/thesis content.

## 2026-06-09 - Codex

**Goal:** Create shared handoff memory for multiple AI agents.
**Summary:** Added repo-level handoff files so Codex, Gemini, Claude, or another agent can understand current status before continuing.
**Files:** `AGENT_HANDOFF.md`, `AGENT_LOG.md`, `NEXT_STEPS.md`, `.agent-logs/codex.md`, `.agent-logs/gemini.md`, `.agent-logs/claude.md`.
**Commands:** None required for repo files.
**Next:** Ask future agents to read the handoff files before making changes.

## 2026-06-09 - Codex

**Goal:** Make fresh agents discover and use the handoff log automatically.
**Summary:** Added agent onboarding instruction files for general AI agents, Claude, and Gemini. Updated the handoff summary to mention these files.
**Files:** `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `AGENT_HANDOFF.md`, `AGENT_LOG.md`, `.agent-logs/codex.md`.
**Commands:** None required.
**Next:** For any new agent, start by reading `AGENTS.md` or the agent-specific instruction file.

## 2026-06-09 - Codex

**Goal:** Check whether the newly provided dataset is visible in the repo.
**Summary:** Found `archive/` containing VisDrone original-format splits. Train, val, and test-dev counts match the already prepared `data/raw` counts; test-challenge has images only.
**Files:** `archive/VisDrone.yaml`, `AGENT_HANDOFF.md`, `AGENT_LOG.md`.
**Commands:** `Get-ChildItem archive` -> found VisDrone train/val/test-dev/test-challenge; split counts confirmed.
**Next:** If retraining from archive, run `scripts/prepare_visdrone.py --source archive --overwrite`, then rebuild caches and train DQN.

## 2026-06-10 - User / Codex

**Goal:** Train the DQN model from the prepared train caches.
**Summary:** Full training completed for 30,000 episodes. Final log line reached episode 30000, and checkpoints were saved.
**Files:** `runs/dqn/best.pt`, `runs/dqn/last.pt`, `runs/dqn/train_log.csv`, `AGENT_HANDOFF.md`, `AGENT_LOG.md`, `.agent-logs/codex.md`.
**Commands:** `python scripts/train.py --split train --episodes 30000` -> completed; `runs/dqn/last.pt` updated at the end of training.
**Next:** Run `python scripts/infer.py --split val` or a limited validation smoke test with `--limit 20`.

## 2026-06-10 - Codex

**Goal:** Check whether the `tvtuan` conda environment and RL-SAHI code are currently using GPU/CUDA.
**Summary:** Confirmed the machine has an NVIDIA RTX 5060 Laptop GPU visible to `nvidia-smi`, but `conda run -n tvtuan` reports `torch 2.8.0+cpu`, `torch.cuda.is_available() == False`, `torch.version.cuda == None`, and `device_count == 0`. Code paths auto-select CUDA only when PyTorch reports it available, so current runs in this env fall back to CPU unless a CUDA-enabled PyTorch build is installed.
**Files:** Inspected `configs/detection.yaml`, `configs/inference.yaml`, `configs/rl.yaml`, `src/rl_sahi/rl/trainer.py`, `src/rl_sahi/inference/pipeline.py`, and `src/rl_sahi/detection/yolo.py`; no source code changes.
**Commands:** `conda run -n tvtuan python -c "... torch.cuda ..."` -> CPU-only torch; `nvidia-smi` -> GPU visible; `conda list -n tvtuan torch` -> torch packages installed from pip; `conda run -n tvtuan pip show torch torchvision torchaudio ultralytics` -> package versions checked. Importing `ultralytics` in the sandbox hit an AppData permission issue unrelated to CUDA.
**Next:** Install a CUDA-enabled PyTorch build in `tvtuan` before expecting RL-SAHI training/inference to use GPU.

## 2026-06-10 - Codex

**Goal:** Make conda env `tvtuan` use GPU by default for PyTorch/RL-SAHI code.
**Summary:** Upgraded the torch stack in `tvtuan` from CPU-only PyTorch to CUDA-enabled PyTorch. Verification now reports `torch 2.11.0+cu128`, `torch.cuda.is_available() == True`, CUDA runtime `12.8`, one GPU, and `NVIDIA GeForce RTX 5060 Laptop GPU`; a test tensor was allocated on `cuda:0`. Ultralytics import also succeeded when run with permissions to read its AppData settings file.
**Files:** Updated handoff/log files only; no RL-SAHI source changes.
**Commands:** `conda run -n tvtuan python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` -> installed `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `torchaudio 2.11.0+cu128`; CUDA verification command -> passed; Ultralytics import verification -> passed; `pip check` -> reported existing unrelated package conflicts and new conflicts for packages such as `unsloth` and `whisperx` that expect older torch versions.
**Next:** For RL-SAHI, run scripts after `conda activate tvtuan`; for unrelated workloads like `whisperx`/`unsloth`, consider a separate environment or pinning a compatible torch version.

## 2026-06-10 - Codex

**Goal:** Fix validation inference state dimension mismatch.
**Summary:** Diagnosed Ultralytics first-predict warmup causing duplicated feature activations in `FeatureCollector`, producing a 2048-dim feature for the first image while the DQN checkpoint expects 1024. Updated `FeatureCollector` to keep only the latest activation per requested layer, rebuilt the first val cache, and confirmed validation inference works on one image.
**Files:** `src/rl_sahi/detection/features.py`, `data/cache/detections/val/0000001_02999_d_0000005.npz`, `AGENT_HANDOFF.md`, `AGENT_LOG.md`, `.agent-logs/codex.md`.
**Commands:** `python scripts/detect.py --split val --limit 1 --overwrite` -> passed; `python scripts/infer.py --split val --limit 1` -> passed.
**Next:** Run full validation inference with `python scripts/infer.py --split val`.

## 2026-06-10 - Codex

**Goal:** Add class names to inference visualization images.
**Summary:** Updated visualization to draw `class name + confidence` labels on each detection box. Class names are read from `YOLO.names` when available, with a VisDrone fallback list.
**Files:** `src/rl_sahi/inference/visualize.py`, `src/rl_sahi/inference/pipeline.py`, `runs/infer/visualizations/0000001_02999_d_0000005.jpg`, `AGENT_HANDOFF.md`, `AGENT_LOG.md`, `.agent-logs/codex.md`.
**Commands:** `python scripts/infer.py --split val --limit 1` -> passed; visualization now includes labels.
**Next:** Run full validation inference to regenerate all visualization images with class labels.

## 2026-06-10 - Codex

**Goal:** Diagnose wrong detection class names and prepare a VisDrone detector workflow.
**Summary:** Confirmed default config still uses COCO pretrained `yolo11s.pt`, which explains labels such as `cell phone`, `potted plant`, and `boat`. Added a VisDrone YOLO dataset config, a separate RL-SAHI config that points at the existing VisDrone-trained YOLO weight, and a YOLO fine-tuning script. Smoke-tested one validation cache with the VisDrone detector; output classes are in the 0-9 VisDrone range and feature dimension is 512, so DQN caches/checkpoints must be rebuilt for this detector.
**Files:** `configs/visdrone_yolo.yaml`, `configs/paths_visdrone.yaml`, `configs/default_visdrone.yaml`, `scripts/train_yolo.py`, `data/cache_visdrone/detections/val/0000001_02999_d_0000005.npz`.
**Commands:** `python -m py_compile scripts/train_yolo.py` -> passed; `python scripts/detect.py --config configs/default_visdrone.yaml --split val --limit 1 --overwrite` -> passed; cache inspection showed `feature_dim=512`, VisDrone class IDs, and 467 boxes.
**Next:** Rebuild VisDrone train/val caches with `--config configs/default_visdrone.yaml`, train a new DQN into `runs/dqn_visdrone`, then run inference with the same config.

## 2026-06-10 - User

**Goal:** Train the DQN using the VisDrone detector workflow.
**Summary:** User completed a 30,000-episode DQN training run for the VisDrone workflow. The final episode reached `ep=30000/30000`, epsilon was `0.050`, and the best checkpoint was saved.
**Files:** `runs/dqn_visdrone/best.pt`.
**Commands:** `python scripts/train.py --config configs/default_visdrone.yaml --split train --episodes 30000` -> completed; best checkpoint saved to `D:\RL-SAHI\runs\dqn_visdrone\best.pt`.
**Next:** Run VisDrone validation inference with `python scripts/infer.py --config configs/default_visdrone.yaml --split val --limit 10`, then full val inference if the visual labels look correct.

## 2026-06-10 - Codex

**Goal:** Prepare the project for GitHub upload by ignoring image/generated files and adding README guidance.
**Summary:** Expanded `.gitignore` to exclude image/media files, datasets, generated caches, outputs, experiment runs, and model artifacts while keeping the existing `yolo11s.pt` exception. Added a root `README.md` in Vietnamese with project overview, environment/GPU check, data preparation, VisDrone workflow commands, default COCO workflow notes, and safer Git upload instructions.
**Files:** `.gitignore`, `README.md`, `AGENT_HANDOFF.md`, `AGENT_LOG.md`, `NEXT_STEPS.md`, `.agent-logs/codex.md`.
**Commands:** `git check-ignore -v data/raw/images/train/example.jpg` -> ignored via `data/raw/`; `git check-ignore -v runs/infer_visdrone/visualizations/example.jpg` -> ignored via `runs/`; `git check-ignore -v data/cache_visdrone/detections/val/example.npz` -> ignored via `data/cache_*/`; `git rev-parse --show-toplevel` -> reported `D:/`, so README warns about the parent repo.
**Next:** Before pushing, stage only intended project files from `D:\RL-SAHI` and verify with `git status --short`.

## 2026-06-10 - Codex

**Goal:** Inspect the added `EDA Data` folder.
**Summary:** Read the main EDA reports and metrics under `EDA Data/adaptive_rl_sahi`. The data report shows 457,066 labels across train/val/test, with 285,028 small labels (62.36%). The full-image YOLO baseline has low recall, especially for small objects (small recall 0.0446). The SAHI baseline improves overall recall to 0.2902 and small recall to 0.1583, but with lower precision, supporting adaptive slicing as a justified technical direction.
**Files:** `EDA Data/adaptive_rl_sahi/README.md`, `outputs/final/data_research_report.md`, `outputs/final/baseline_metrics.json`, `outputs/final/sahi_metrics.json`, `outputs/final/sahi_test_predata_summary.json`, `outputs/final/test_test_analysis_report.md`.
**Commands:** `Get-Content` on the main EDA reports and metrics -> completed.
**Next:** Use these EDA numbers in thesis/slide justification and technical discussion for adaptive ROI/SAHI.

## 2026-06-10 - Codex

**Goal:** Add same-model technical tuning configs for better small-object recall.
**Summary:** Added a separate VisDrone tuned config stack that keeps the same YOLO weight but changes inference calibration and RL environment/reward settings. The tuned config lowers output confidence to 0.15, raises merge IoU to 0.6, allows more slice attempts, uses smaller/scale-aware ROI bounds, allows up to 10 slices, and increases hard-region rewards/empty/overlap penalties. A one-image inference smoke test using the previous checkpoint passed and wrote to the tuned output folder.
**Files:** `configs/paths_visdrone_tuned.yaml`, `configs/inference_visdrone_tuned.yaml`, `configs/rl_visdrone_tuned.yaml`, `configs/default_visdrone_tuned.yaml`.
**Commands:** Config load check -> passed; `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --checkpoint runs\dqn_visdrone\best.pt --split val --limit 1` -> passed with 75 boxes and 2 slices.
**Next:** Train a tuned DQN with `python scripts/train.py --config configs/default_visdrone_tuned.yaml --split train --episodes 30000`, then run tuned validation inference without checkpoint override.

## 2026-06-10 - Codex

**Goal:** Improve slice coverage when DQN repeats/overlaps ROIs.
**Summary:** Changed inference so an overlapping DQN ROI is rejected and inference continues instead of stopping all slice attempts. Added a density fallback ROI generator that fills remaining slice slots from small/low-confidence detection clusters when the DQN stalls. On `0000006_00611_d_0000002.jpg`, tuned inference changed from 87 boxes with 3 slices to 158 boxes with 10 slices; metadata now records `density_fallback` slices.
**Files:** `src/rl_sahi/inference/pipeline.py`, `runs/infer_visdrone_tuned/metadata/0000006_00611_d_0000002.json`, `runs/infer_visdrone_tuned/visualizations/0000006_00611_d_0000002.jpg`.
**Commands:** `python -m py_compile src\rl_sahi\inference\pipeline.py` -> passed; `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --image ...0000006_00611_d_0000002.jpg --split test --no-cache` -> passed with 158 boxes and 10 slices.
**Next:** Visually inspect the tuned image for false positives; if too noisy, raise `output_conf` slightly or set `min_slice_detections` higher.

## 2026-06-10 - Codex

**Goal:** Reduce visual clutter from too many slices.
**Summary:** Added an inference-time `max_slices` override so visualization can be limited without retraining the DQN checkpoint. Tuned config now uses `max_slices=5`, `output_conf=0.18`, `min_slice_detections=3`, and `max_slice_attempts=12`. The same test image now produces 120 boxes with 5 slices instead of 158 boxes with 10 slices.
**Files:** `configs/inference_visdrone_tuned.yaml`, `configs/rl_visdrone_tuned.yaml`, `src/rl_sahi/inference/config.py`, `src/rl_sahi/inference/pipeline.py`, `scripts/infer.py`.
**Commands:** `python -m py_compile src\rl_sahi\inference\config.py src\rl_sahi\inference\pipeline.py scripts\infer.py` -> passed; tuned single-image inference -> passed with 120 boxes and 5 slices.
**Next:** Inspect the image; if still cluttered, reduce `infer.max_slices` to 3 or raise `output_conf` to 0.20.

## 2026-06-10 - Codex

**Goal:** Export per-image predicted class statistics.
**Summary:** Added a summary script that reads inference detection `.txt` files and writes a human-readable class-count report per image, including total boxes plus full-image vs slice-origin counts. Generated the tuned report for current outputs.
**Files:** `scripts/summarize_predictions.py`, `runs/infer_visdrone_tuned/class_summary.txt`.
**Commands:** `python -m py_compile scripts\summarize_predictions.py` -> passed; `python scripts\summarize_predictions.py --pred-dir runs\infer_visdrone_tuned\detections --out runs\infer_visdrone_tuned\class_summary.txt` -> wrote report for 11 images.
**Next:** Re-run the summary script after full validation inference to refresh the report.

## 2026-06-10 - Codex

**Goal:** Store one class-summary text file per inferred image.
**Summary:** Updated the prediction summary script to also write `class_summaries/<image_id>.txt` files. Each per-image file includes total boxes, class counts, full-image vs slice-origin counts, and a tabular class breakdown.
**Files:** `scripts/summarize_predictions.py`, `runs/infer_visdrone_tuned/class_summaries/`.
**Commands:** `python scripts\summarize_predictions.py --pred-dir runs\infer_visdrone_tuned\detections --out runs\infer_visdrone_tuned\class_summary.txt --per-image-dir runs\infer_visdrone_tuned\class_summaries` -> wrote 11 per-image summary files.
**Next:** Keep `runs/infer_visdrone_tuned/class_summaries` beside visualizations for easy inspection.

## 2026-06-10 - Codex

**Goal:** Make visualization easier to read.
**Summary:** Removed detection text labels/confidence from visualization images and changed detection boxes to use a fixed color per VisDrone class. ROI slice boxes remain drawn separately. Regenerated the tuned visualization for `0000006_00611_d_0000002.jpg`.
**Files:** `src/rl_sahi/inference/visualize.py`, `runs/infer_visdrone_tuned/visualizations/0000006_00611_d_0000002.jpg`.
**Commands:** `python -m py_compile src\rl_sahi\inference\visualize.py scripts\infer.py` -> passed; tuned single-image inference -> passed with 120 boxes and 5 slices.
**Next:** If a legend is needed, use the per-image class summary `.txt` instead of drawing text on the image.

## 2026-06-10 - Codex

**Goal:** Add a class-color legend for text-free visualizations.
**Summary:** Updated `summarize_predictions.py` to write `class_color_legend.txt` beside the tuned inference outputs. The legend maps each VisDrone class to its fixed visualization color and RGB hex value.
**Files:** `scripts/summarize_predictions.py`, `runs/infer_visdrone_tuned/class_color_legend.txt`.
**Commands:** `python scripts\summarize_predictions.py ... --legend-out runs\infer_visdrone_tuned\class_color_legend.txt` -> wrote the legend.
**Next:** Keep text off visualization images and use the legend plus per-image summary files for class interpretation.

## 2026-06-10 - Codex

**Goal:** Restore class names on visualization boxes.
**Summary:** Updated visualization to show only the class name on each detection box, without confidence scores. Per-image summary `.txt` files remain the main detailed count output. Regenerated the tuned test image.
**Files:** `src/rl_sahi/inference/visualize.py`, `runs/infer_visdrone_tuned/visualizations/0000006_00611_d_0000002.jpg`, `runs/infer_visdrone_tuned/class_summaries/`.
**Commands:** `python -m py_compile src\rl_sahi\inference\visualize.py scripts\infer.py scripts\summarize_predictions.py` -> passed; tuned single-image inference -> passed with 120 boxes and 5 slices.
**Next:** Use class-name-only visualization plus per-image summaries for inspection.

## 2026-06-10 - Codex

**Goal:** Reduce duplicate boxes and wrong-class overlaps in tuned inference.
**Summary:** Added class-wise confidence thresholds and class-agnostic/source-aware NMS to post-processing. The tuned image `0000006_00611_d_0000002.jpg` now reduces from 155 pre-filter boxes to 85 final boxes while keeping 5 slices. Metadata records post-processing counts.
**Files:** `configs/inference_visdrone_tuned.yaml`, `src/rl_sahi/inference/config.py`, `src/rl_sahi/inference/pipeline.py`, `scripts/infer.py`, `runs/infer_visdrone_tuned/metadata/0000006_00611_d_0000002.json`.
**Commands:** `python -m py_compile src\rl_sahi\inference\config.py src\rl_sahi\inference\pipeline.py scripts\infer.py` -> passed; tuned single-image inference -> passed with 85 boxes and 5 slices; summary script rerun passed.
**Next:** Visually inspect whether duplicate and wrong-class boxes improved. If duplicates remain, lower `agnostic_nms_iou`; if true objects disappear, raise it slightly.

## 2026-06-10 - Codex

**Goal:** Smoke test tuned inference on 20 validation images.
**Summary:** Ran tuned inference on the first 20 validation images after post-processing refinement. All 20 completed successfully, mostly using 5 slices per image. Refreshed class summaries; the output folder currently summarizes 21 images because it also contains the earlier single test image.
**Files:** `runs/infer_visdrone_tuned/visualizations/`, `runs/infer_visdrone_tuned/detections/`, `runs/infer_visdrone_tuned/metadata/`, `runs/infer_visdrone_tuned/class_summary.txt`, `runs/infer_visdrone_tuned/class_summaries/`.
**Commands:** `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --split val --limit 20` -> passed; `python scripts\summarize_predictions.py ...` -> refreshed combined and per-image summaries.
**Next:** Inspect several visualizations and tune `agnostic_nms_iou`, `output_conf`, or class thresholds if duplicates/wrong classes remain.
