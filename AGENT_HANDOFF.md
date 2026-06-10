# Agent Handoff

## Current Goal
Maintain shared project context for AI agents working on this RL-SAHI repository. The current project focuses on adaptive slicing for small object detection using YOLO, SAHI-style inference, and a DQN agent for ROI selection.

## Current Status
- Done: Project structure has been reviewed; train/val detection and hard-region caches exist for the original COCO `yolo11s.pt` workflow; inference/cache generation were tested; full DQN training for 30,000 episodes completed successfully for that original detector. The VisDrone detector workflow has also completed a 30,000-episode DQN training run and saved `runs/dqn_visdrone/best.pt`.
- In progress: Post-training validation inference for the VisDrone detector workflow.
- Blocked: No active blocker.

## Last Completed Work
- 2026-06-10 Codex: Prepared GitHub upload support by expanding `.gitignore` for images/datasets/generated outputs and adding a root Vietnamese `README.md` with setup, workflow, and Git instructions.
- 2026-06-10 Codex: Installed CUDA-enabled PyTorch in conda env `tvtuan`; verification now sees `torch 2.11.0+cu128`, CUDA available, and GPU `NVIDIA GeForce RTX 5060 Laptop GPU`.
- 2026-06-10 Codex: Checked GPU/CUDA status for conda env `tvtuan`; machine GPU is visible via `nvidia-smi`, but env has CPU-only PyTorch (`torch 2.8.0+cpu`) so current code falls back to CPU.
- 2026-06-09 Codex: Explained project testing workflow, verified sample commands, drafted thesis text for RL input data, and planned a 5-6 slide presentation.
- 2026-06-09 Codex: Set up cross-agent handoff logging files in the repo.
- 2026-06-09 Codex: Added repo-level onboarding instruction files for Codex-compatible agents, Claude, and Gemini.
- 2026-06-09 Codex: Inspected `archive/` dataset and confirmed it contains VisDrone original-format train/val/test-dev/test-challenge splits.
- 2026-06-10 User/Codex: Completed DQN training for 30,000 episodes; `runs/dqn/best.pt`, `runs/dqn/last.pt`, and `runs/dqn/train_log.csv` were updated.
- 2026-06-10 Codex: Fixed validation inference state-dimension mismatch by making YOLO feature hooks keep only the latest activation per layer; `python scripts/infer.py --split val --limit 1` now passes.
- 2026-06-10 Codex: Added class-name and confidence labels to inference visualization images.
- 2026-06-10 Codex: Diagnosed wrong labels as a detector-weight issue: default `yolo11s.pt` is COCO pretrained, while VisDrone requires 10 classes. Added VisDrone-specific YOLO config, RL-SAHI config, and a YOLO training script.
- 2026-06-10 User: Completed VisDrone DQN training for 30,000 episodes; best checkpoint saved to `D:\RL-SAHI\runs\dqn_visdrone\best.pt`.

## Important Decisions
- Use repo-root Markdown files as the shared memory layer so Codex, Gemini, Claude, and other agents can all read them.
- Codex global skill installed at `C:\Users\ADMIN\.codex\skills\agent-handoff-log` for future Codex sessions.
- Add `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` so fresh agents see handoff instructions immediately when entering the repo.
- Keep detailed transient chat context out of the log; record only durable project facts, changed files, commands, decisions, blockers, and next steps.
- For RL-SAHI, treat `data/cache/detections/<split>` and `data/cache/hard_regions/<split>` as RL input caches, not raw images directly.
- Do not fix wrong COCO labels by remapping display names. The detector itself must be VisDrone-trained; otherwise class predictions remain semantically wrong.
- Use `configs/default_visdrone.yaml` for the VisDrone detector workflow. It writes to `data/cache_visdrone`, `runs/dqn_visdrone`, and `runs/infer_visdrone` so it does not mix with the old COCO-detector caches.

## Files Changed Recently
- `.gitignore`: Expanded to ignore image/media files, datasets, caches, outputs, runs, and generated model artifacts.
- `README.md`: Root project guide for GitHub, environment setup, data preparation, VisDrone workflow, and Git upload commands.
- `AGENT_HANDOFF.md`: Shared current state and handoff summary.
- `AGENT_LOG.md`: Chronological multi-agent activity log.
- `NEXT_STEPS.md`: Prioritized continuation tasks.
- `AGENTS.md`: General AI-agent onboarding instructions.
- `CLAUDE.md`: Claude-specific handoff instructions.
- `GEMINI.md`: Gemini-specific handoff instructions.
- `runs/dqn/best.pt`: Best DQN checkpoint from 30,000-episode training.
- `runs/dqn/last.pt`: Final DQN checkpoint from 30,000-episode training.
- `runs/dqn/train_log.csv`: Full training log through episode 30,000.
- `runs/dqn_visdrone/best.pt`: Best DQN checkpoint from 30,000-episode VisDrone detector training.
- `src/rl_sahi/detection/features.py`: Fixed feature collection so Ultralytics first-predict warmup does not double feature vector length.
- `src/rl_sahi/inference/visualize.py`: Draws class labels and confidence scores on detection boxes.
- `src/rl_sahi/inference/pipeline.py`: Passes YOLO model class names, scores, and classes into visualization.
- `configs/visdrone_yolo.yaml`: Ultralytics dataset config for `D:/RL-SAHI/data/raw` with the 10 VisDrone classes.
- `configs/paths_visdrone.yaml`: VisDrone detector paths using the existing fine-tuned YOLO weight and separate cache/output roots.
- `configs/default_visdrone.yaml`: RL-SAHI config entrypoint for the VisDrone detector workflow.
- `scripts/train_yolo.py`: Fine-tunes YOLO on the prepared VisDrone dataset.
- `data/cache_visdrone/detections/val/0000001_02999_d_0000005.npz`: Smoke-test detection cache from the VisDrone detector.
- `data/cache/detections/val/0000001_02999_d_0000005.npz`: Rebuilt after feature-collector fix.
- `.agent-logs/codex.md`: Codex-specific activity log.
- `.agent-logs/gemini.md`: Placeholder for Gemini-specific activity log.
- `.agent-logs/claude.md`: Placeholder for Claude-specific activity log.

## Commands / Tests Run Recently
- `git check-ignore -v data/raw/images/train/example.jpg`: Confirmed raw dataset images are ignored.
- `git check-ignore -v runs/infer_visdrone/visualizations/example.jpg`: Confirmed generated visualization images are ignored.
- `git check-ignore -v data/cache_visdrone/detections/val/example.npz`: Confirmed generated VisDrone cache files are ignored.
- `git rev-parse --show-toplevel`: Reported `D:/`, meaning this folder is currently inside a parent Git repo; use careful staging or initialize a separate repo for GitHub upload.
- `conda run -n tvtuan python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`: Installed CUDA-enabled torch stack.
- `conda run -n tvtuan python -c "... torch.cuda ..."`: Passed after upgrade; reported `torch 2.11.0+cu128`, CUDA `12.8`, GPU count 1, and tensor allocation on `cuda:0`.
- `conda run -n tvtuan python -c "import torch; from ultralytics import YOLO; ..."`: Passed with elevated permissions; Ultralytics import OK.
- `conda run -n tvtuan python -m pip check`: Reported dependency conflicts, including `unsloth`/`unsloth-zoo` and `whisperx` expecting older torch versions; RL-SAHI CUDA verification still passed.
- `conda run -n tvtuan python -c "... torch.cuda ..."`: Confirmed `torch 2.8.0+cpu`, CUDA unavailable, CUDA version `None`, and device count 0.
- `nvidia-smi`: Confirmed NVIDIA GeForce RTX 5060 Laptop GPU is visible at driver level.
- `conda list -n tvtuan torch` and `conda run -n tvtuan pip show torch torchvision torchaudio ultralytics`: Confirmed installed torch stack comes from pip and is CPU-only.
- `python scripts/infer.py --split train --limit 3`: Passed; produced detections for 3 train images.
- `python scripts/detect.py --split val --limit 5`: Passed; wrote 5 validation detection caches.
- `python scripts/hard_region.py --split val --limit 5`: Passed; wrote 5 validation hard-region caches.
- `python scripts/infer.py --split val --limit 5`: Passed; produced detections for 5 validation images.
- `python scripts/train.py --split train --episodes 5 --limit 20`: Passed; saved DQN checkpoint.
- `python scripts/train.py --split train --episodes 30000`: Completed; final logged episode 30000, best checkpoint saved to `runs/dqn/best.pt`.
- `Get-ChildItem archive/...`: Confirmed archive has 6471 train images/annotations, 548 val images/annotations, 1610 test-dev images/annotations, and 1580 test-challenge images without annotations.
- `python scripts/detect.py --split val --limit 1 --overwrite`: Passed; rebuilt first val detection cache with 1024-dim feature.
- `python scripts/infer.py --split val --limit 1`: Passed; first validation image produced 21 boxes with 3 slices.
- `python scripts/infer.py --split val --limit 1`: Passed after visualization update; generated class-name labels on `runs/infer/visualizations/0000001_02999_d_0000005.jpg`.
- `python -c "from ultralytics import YOLO; ..."`: Loaded `EDA Data/adaptive_rl_sahi/baseline/training_runs/yolo11n_visdrone/weights/best.pt`; model names are the correct 10 VisDrone classes.
- `python -m py_compile scripts/train_yolo.py`: Passed.
- `python scripts/detect.py --config configs/default_visdrone.yaml --split val --limit 1 --overwrite`: Passed; wrote one cache under `data/cache_visdrone/detections/val`.
- `python -c "import numpy as np; ..."`: Confirmed the VisDrone cache has `feature_dim=512`, classes in 0-9, and 467 boxes.

## Known Issues / Risks
- `D:\RL-SAHI` is inside a parent Git repo rooted at `D:/`; do not run broad `git add .` from the parent unless that is intentional.
- The `tvtuan` conda env now has CUDA-enabled PyTorch and should use GPU for RL-SAHI code that auto-selects CUDA.
- `pip check` reports package conflicts in `tvtuan`, including `unsloth`/`unsloth-zoo` and `whisperx` requiring older torch versions. Use a separate env for those workloads if needed.
- Importing `ultralytics` under the restricted sandbox can hit a permission error reading `C:\Users\ADMIN\AppData\Roaming\Ultralytics\settings.json`; running with normal user permissions succeeded.
- No formal test suite is present.
- No quantitative mAP/Recall evaluation script has been confirmed.
- Existing `runs/dqn/best.pt` was trained with the original COCO `yolo11s.pt` detector/cache. It should not be used with `configs/default_visdrone.yaml`, because the VisDrone YOLO feature vector is 512-dim and produces a different DQN state.
- `git status` may report unrelated files from a parent repository; be careful before staging or committing.
- Long training with full data may take significant time.
- `archive/VisDrone2019-DET-test-challenge` has images but no labels, so it is not suitable for hard-region reward/training without annotations.

## Next Recommended Steps
1. Run a limited validation inference smoke test with `python scripts/infer.py --config configs/default_visdrone.yaml --split val --limit 10`.
2. Inspect `runs/infer_visdrone/visualizations` and confirm labels are VisDrone classes.
3. Run full validation inference with `python scripts/infer.py --config configs/default_visdrone.yaml --split val`.
4. Add or run quantitative evaluation for YOLO full-image vs adaptive slicing.
5. Prepare final presentation slides from the existing 5-6 slide outline.
6. Keep this handoff updated whenever another agent completes work.

## Do Not Touch Without Approval
- Large dataset files under `data/raw/`.
- Model checkpoints under `runs/dqn/`.
- YOLO weights `yolo11s.pt`.
- Any unrelated files outside `D:\RL-SAHI`.
