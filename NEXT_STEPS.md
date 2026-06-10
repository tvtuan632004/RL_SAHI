# Next Steps

1. Train the tuned same-model DQN: `python scripts/train.py --config configs/default_visdrone_tuned.yaml --split train --episodes 30000`.
2. Run tuned validation smoke test: `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --split val --limit 10`.
3. Compare `runs/infer_visdrone/visualizations` against `runs/infer_visdrone_tuned/visualizations`.
4. Run full tuned validation inference with `python scripts/infer.py --config configs/default_visdrone_tuned.yaml --split val`.
5. Add or run mAP/Recall evaluation for adaptive slicing results.
6. Compare adaptive RL slicing against YOLO full-image inference.
7. Compare adaptive RL slicing against fixed SAHI slicing if available.
8. Prepare final presentation slides from the 5-6 slide outline.
9. Before GitHub upload, stage only intended `D:\RL-SAHI` files and verify ignored image/data/output files are not staged.
10. Keep `tvtuan` dependency conflicts in mind if using unrelated packages such as `whisperx` or `unsloth`.
11. Update `AGENT_HANDOFF.md` and `AGENT_LOG.md` after each work session.
12. When switching agents, ask the next agent to read `AGENTS.md` first.
