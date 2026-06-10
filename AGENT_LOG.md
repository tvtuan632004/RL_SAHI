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
