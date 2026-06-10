from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from rl_sahi.common.boxes import clip_boxes
from rl_sahi.common.cache import (
    DetectionCache,
    detection_cache_is_current,
    detection_cache_path,
    load_detection_cache,
    save_detection_cache,
)
from rl_sahi.common.config import ProjectConfig, load_default_config
from rl_sahi.detection.yolo import detect_one_image, load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crop
from rl_sahi.inference.merge import class_aware_nms, save_prediction_txt
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.inference.visualize import save_inference_visual
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig


def get_initial_detection(
    model: YOLO,
    image_path: Path,
    weights_imgsz: int,
    full_conf: float,
    full_iou: float,
    max_det: int,
    device: str | None,
    feature_layers: tuple[int, ...],
    aux_grid_size: int,
    spatial_feature_channels: int,
    cache_root: Path | str | None = None,
    split: str | None = None,
    use_cache: bool = True,
) -> DetectionCache:
    if cache_root is not None and split is not None:
        cache_path = detection_cache_path(cache_root, split, image_path)
        if use_cache and detection_cache_is_current(cache_path):
            return load_detection_cache(cache_path)
        det = detect_one_image(
            model=model,
            image_path=image_path,
            imgsz=weights_imgsz,
            conf=full_conf,
            iou=full_iou,
            max_det=max_det,
            device=device,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        save_detection_cache(cache_path, det)
        return det
    return detect_one_image(
        model=model,
        image_path=image_path,
        imgsz=weights_imgsz,
        conf=full_conf,
        iou=full_iou,
        max_det=max_det,
        device=device,
        feature_layers=feature_layers,
        aux_grid_size=aux_grid_size,
        spatial_feature_channels=spatial_feature_channels,
    )


class AdaptiveSahiInferencer:
    def __init__(self, weights: Path, checkpoint: Path, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device_t = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy, checkpoint_data = load_policy(checkpoint, self.device_t)
        self.env_cfg = checkpoint_data["env_cfg_obj"]
        self.state_cfg = checkpoint_data.get("state_cfg_obj", StateConfig())
        self.yolo = load_yolo(weights, device=cfg.device)

    def infer_image(
        self,
        image_path: Path,
        out_dir: Path,
        cache_root: Path | None = None,
        split: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        cfg = self.cfg
        det = get_initial_detection(
            model=self.yolo,
            image_path=image_path,
            weights_imgsz=cfg.full_imgsz,
            full_conf=cfg.full_conf,
            full_iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
            feature_layers=cfg.feature_layers,
            aux_grid_size=self.state_cfg.grid_size,
            spatial_feature_channels=self.state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )

        return _infer_with_loaded(
            image_path=image_path,
            out_dir=out_dir,
            yolo=self.yolo,
            class_names=getattr(self.yolo, "names", None),
            policy=self.policy,
            device_t=self.device_t,
            env_cfg=self.env_cfg,
            state_cfg=self.state_cfg,
            det=det,
            cfg=cfg,
        )


def _infer_with_loaded(
    image_path: Path,
    out_dir: Path,
    yolo: YOLO,
    class_names,
    policy,
    device_t: torch.device,
    env_cfg,
    state_cfg: StateConfig,
    det: DetectionCache,
    cfg: InferenceConfig,
) -> dict:

    accepted_rois: list[np.ndarray] = []
    rejected_rois: list[np.ndarray] = []
    attempted_rois: list[np.ndarray] = []
    slice_boxes_all: list[np.ndarray] = []
    slice_scores_all: list[np.ndarray] = []
    slice_classes_all: list[np.ndarray] = []
    slice_sources_all: list[np.ndarray] = []
    slice_meta: list[dict] = []

    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)
    for attempt_idx in range(1, max_attempts + 1):
        if len(accepted_rois) >= env_cfg.max_slices:
            break
        previous_arr = (
            np.stack(attempted_rois).astype(np.float32)
            if attempted_rois
            else np.zeros((0, 4), dtype=np.float32)
        )
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=previous_arr)
        roi, actions, info = rollout_one_slice(policy, env, device_t)
        if info.get("stop_due_to_old_overlap", False):
            slice_meta.append(
                {
                    "attempt_index": attempt_idx,
                    "slice_index": None,
                    "accepted": False,
                    "rejection_reason": "old_slice_overlap",
                    "roi": [float(x) for x in roi.tolist()],
                    "actions": actions,
                    "steps": len(actions),
                    "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                    "detections": 0,
                }
            )
            break

        boxes_i, scores_i, classes_i = run_yolo_on_crop(
            yolo,
            image_path,
            roi,
            imgsz=cfg.slice_imgsz,
            conf=cfg.output_conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
        )
        attempted_rois.append(roi)
        accepted = int(len(boxes_i)) >= int(cfg.min_slice_detections)
        rejection_reason = None if accepted else ("empty_slice" if len(boxes_i) == 0 else "low_detection_count")
        slice_index = None
        if accepted:
            accepted_rois.append(roi)
            slice_index = len(accepted_rois)
            slice_boxes_all.append(boxes_i)
            slice_scores_all.append(scores_i)
            slice_classes_all.append(classes_i)
            slice_sources_all.append(np.ones((len(boxes_i),), dtype=np.int32))
        else:
            rejected_rois.append(roi)
        slice_meta.append(
            {
                "attempt_index": attempt_idx,
                "slice_index": slice_index,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "roi": [float(x) for x in roi.tolist()],
                "actions": actions,
                "steps": len(actions),
                "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                "detections": int(len(boxes_i)),
            }
        )

    full_mask = det.scores >= cfg.output_conf
    full_boxes = det.boxes[full_mask]
    full_scores = det.scores[full_mask]
    full_classes = det.classes[full_mask]

    boxes_parts = [full_boxes] + slice_boxes_all
    scores_parts = [full_scores] + slice_scores_all
    classes_parts = [full_classes] + slice_classes_all
    sources_parts = [np.zeros((len(full_boxes),), dtype=np.int32)] + slice_sources_all

    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    sources = np.concatenate(sources_parts, axis=0) if sources_parts else np.zeros((0,), dtype=np.int32)

    boxes = clip_boxes(boxes, det.image_shape)
    keep = class_aware_nms(boxes, scores, classes, cfg.merge_iou)
    boxes, scores, classes, sources = boxes[keep], scores[keep], classes[keep], sources[keep]

    out_dir = Path(out_dir)
    pred_path = out_dir / "detections" / f"{image_path.stem}.txt"
    viz_path = out_dir / "visualizations" / f"{image_path.stem}.jpg"
    meta_path = out_dir / "metadata" / f"{image_path.stem}.json"
    accepted_rois_array = (
        np.stack(accepted_rois).astype(np.float32) if accepted_rois else np.zeros((0, 4), dtype=np.float32)
    )
    rejected_rois_array = (
        np.stack(rejected_rois).astype(np.float32) if rejected_rois else np.zeros((0, 4), dtype=np.float32)
    )
    save_prediction_txt(pred_path, boxes, scores, classes, sources)
    save_inference_visual(
        image_path,
        boxes,
        scores,
        classes,
        sources,
        accepted_rois_array,
        rejected_rois_array,
        viz_path,
        class_names=class_names,
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "image": str(image_path),
        "num_slices": len(accepted_rois),
        "num_attempts": len(slice_meta),
        "num_rejected_slices": len(rejected_rois),
        "slices": slice_meta,
        "detections": int(len(boxes)),
        "prediction_file": str(pred_path),
        "visualization_file": str(viz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _resolve_project_path(path: Path | str, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def _config_path_or_override(cfg: ProjectConfig, key: str, value: Path | str | None) -> Path:
    if value is None:
        return cfg.path_value(key)
    return _resolve_project_path(value, cfg.root)


def _value_or_config(section: dict, key: str, value, cast):
    raw = section[key] if value is None else value
    return cast(raw)


def _feature_layers_or_config(cfg: ProjectConfig, value: tuple[int, ...] | list[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return cfg.feature_layers("infer")
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def infer_one_image(
    image_path: Path | str,
    weights: Path | str | None = None,
    checkpoint: Path | str | None = None,
    out_dir: Path | str | None = None,
    cache_root: Path | None = None,
    split: str | None = None,
    use_cache: bool | None = None,
    full_imgsz: int | None = None,
    slice_imgsz: int | None = None,
    full_conf: float | None = None,
    output_conf: float | None = None,
    iou: float | None = None,
    merge_iou: float | None = None,
    max_det: int | None = None,
    device: str | None = None,
    feature_layers: tuple[int, ...] | list[int] | str | None = None,
    min_slice_detections: int | None = None,
    max_slice_attempts: int | None = None,
    config: ProjectConfig | Path | str | None = None,
) -> dict:
    project_cfg = config if isinstance(config, ProjectConfig) else load_default_config(config)
    infer_cfg = project_cfg.section("infer")
    image_path = _resolve_project_path(image_path, project_cfg.root)
    weights = _config_path_or_override(project_cfg, "weights", weights)
    checkpoint = _config_path_or_override(project_cfg, "checkpoint", checkpoint)
    out_dir = _config_path_or_override(project_cfg, "infer_out_dir", out_dir)
    cache_root = _config_path_or_override(project_cfg, "cache_root", cache_root)
    use_cache = bool(infer_cfg.get("use_cache", True)) if use_cache is None else bool(use_cache)

    cfg = InferenceConfig(
        full_imgsz=_value_or_config(infer_cfg, "full_imgsz", full_imgsz, int),
        slice_imgsz=_value_or_config(infer_cfg, "slice_imgsz", slice_imgsz, int),
        full_conf=_value_or_config(infer_cfg, "full_conf", full_conf, float),
        output_conf=_value_or_config(infer_cfg, "output_conf", output_conf, float),
        iou=_value_or_config(infer_cfg, "iou", iou, float),
        merge_iou=_value_or_config(infer_cfg, "merge_iou", merge_iou, float),
        max_det=_value_or_config(infer_cfg, "max_det", max_det, int),
        device=device if device is not None else project_cfg.optional_str("infer", "device"),
        feature_layers=_feature_layers_or_config(project_cfg, feature_layers),
        min_slice_detections=_value_or_config(infer_cfg, "min_slice_detections", min_slice_detections, int),
        max_slice_attempts=_value_or_config(infer_cfg, "max_slice_attempts", max_slice_attempts, int),
    )
    inferencer = AdaptiveSahiInferencer(weights=weights, checkpoint=checkpoint, cfg=cfg)
    return inferencer.infer_image(
        image_path=image_path,
        out_dir=out_dir,
        cache_root=cache_root,
        split=split,
        use_cache=use_cache,
    )
