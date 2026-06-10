# Next Steps

1. Run a small VisDrone validation inference smoke test: `python scripts/infer.py --config configs/default_visdrone.yaml --split val --limit 10`.
2. Inspect images under `runs/infer_visdrone/visualizations` to confirm labels are VisDrone classes.
3. Run full validation inference with `python scripts/infer.py --config configs/default_visdrone.yaml --split val`.
4. Add or run mAP/Recall evaluation for adaptive slicing results.
5. Compare adaptive RL slicing against YOLO full-image inference.
6. Compare adaptive RL slicing against fixed SAHI slicing if available.
7. Prepare final presentation slides from the 5-6 slide outline.
8. Before GitHub upload, stage only intended `D:\RL-SAHI` files and verify ignored image/data/output files are not staged.
9. Keep `tvtuan` dependency conflicts in mind if using unrelated packages such as `whisperx` or `unsloth`.
10. Update `AGENT_HANDOFF.md` and `AGENT_LOG.md` after each work session.
11. When switching agents, ask the next agent to read `AGENTS.md` first.
