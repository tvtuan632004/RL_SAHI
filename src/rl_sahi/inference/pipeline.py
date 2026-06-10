from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from rl_sahi.common.boxes import area, box_from_center, centers, clip_boxes, intersection_matrix
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
from rl_sahi.common.nms import nms_numpy
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

    max_slices = int(cfg.max_slices) if cfg.max_slices > 0 else int(env_cfg.max_slices)
    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)
    for attempt_idx in range(1, max_attempts + 1):
        if len(accepted_rois) >= max_slices:
            break
        previous_arr = (
            np.stack(attempted_rois).astype(np.float32)
            if attempted_rois
            else np.zeros((0, 4), dtype=np.float32)
        )
        env = SliceEnv(det, None, env_cfg=env_cfg, state_cfg=state_cfg, previous_rois=previous_arr)
        roi, actions, info = rollout_one_slice(policy, env, device_t)
        if info.get("stop_due_to_old_overlap", False):
            attempted_rois.append(roi)
            rejected_rois.append(roi)
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
            continue

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

    fallback_rois = _density_fallback_rois(
        det=det,
        env_cfg=env_cfg,
        state_cfg=state_cfg,
        previous_rois=np.stack(attempted_rois).astype(np.float32)
        if attempted_rois
        else np.zeros((0, 4), dtype=np.float32),
        max_candidates=max_attempts,
    )
    for fallback_idx, roi in enumerate(fallback_rois, start=1):
        if len(accepted_rois) >= max_slices:
            break
        if _max_roi_overlap(roi, np.asarray(attempted_rois, dtype=np.float32)) >= env_cfg.old_slice_overlap_threshold:
            continue

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
                "attempt_index": len(slice_meta) + 1,
                "slice_index": slice_index,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "roi": [float(x) for x in roi.tolist()],
                "actions": ["density_fallback"],
                "steps": 0,
                "old_slice_overlap": float(_max_roi_overlap(roi, np.asarray(attempted_rois[:-1], dtype=np.float32))),
                "detections": int(len(boxes_i)),
                "fallback_index": fallback_idx,
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
    pre_filter_count = int(len(boxes))
    boxes, scores, classes, sources = _apply_class_thresholds(boxes, scores, classes, sources, cfg)
    class_threshold_count = int(len(boxes))
    keep = class_aware_nms(boxes, scores, classes, cfg.merge_iou)
    boxes, scores, classes, sources = boxes[keep], scores[keep], classes[keep], sources[keep]
    class_nms_count = int(len(boxes))
    if cfg.agnostic_nms_iou > 0:
        keep = _source_aware_agnostic_nms(boxes, scores, sources, cfg.agnostic_nms_iou, cfg.slice_score_bonus)
        boxes, scores, classes, sources = boxes[keep], scores[keep], classes[keep], sources[keep]
    agnostic_nms_count = int(len(boxes))

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
        "postprocess": {
            "pre_filter_detections": pre_filter_count,
            "after_class_thresholds": class_threshold_count,
            "after_class_aware_nms": class_nms_count,
            "after_agnostic_nms": agnostic_nms_count,
            "output_conf": float(cfg.output_conf),
            "class_conf_thresholds": _normalized_class_thresholds(cfg),
            "merge_iou": float(cfg.merge_iou),
            "agnostic_nms_iou": float(cfg.agnostic_nms_iou),
            "slice_score_bonus": float(cfg.slice_score_bonus),
        },
        "prediction_file": str(pred_path),
        "visualization_file": str(viz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _normalized_class_thresholds(cfg: InferenceConfig) -> dict[str, float]:
    raw = cfg.class_conf_thresholds or {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        try:
            out[str(int(key))] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _threshold_for_class(class_id: float, cfg: InferenceConfig) -> float:
    thresholds = _normalized_class_thresholds(cfg)
    return float(thresholds.get(str(int(class_id)), cfg.output_conf))


def _apply_class_thresholds(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
    cfg: InferenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return boxes, scores, classes, sources
    thresholds = np.asarray([_threshold_for_class(cls, cfg) for cls in classes], dtype=np.float32)
    keep = np.asarray(scores, dtype=np.float32).reshape(-1) >= thresholds
    return boxes[keep], scores[keep], classes[keep], sources[keep]


def _source_aware_agnostic_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sources: np.ndarray,
    iou_threshold: float,
    slice_score_bonus: float,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    adjusted_scores = np.asarray(scores, dtype=np.float32).reshape(-1).copy()
    adjusted_scores += (np.asarray(sources, dtype=np.int32).reshape(-1) > 0).astype(np.float32) * float(slice_score_bonus)
    keep = nms_numpy(boxes, adjusted_scores, iou_threshold)
    return keep[np.argsort(scores[keep])[::-1]].astype(np.int64)


def _density_fallback_rois(
    det: DetectionCache,
    env_cfg,
    state_cfg: StateConfig,
    previous_rois: np.ndarray,
    max_candidates: int,
) -> list[np.ndarray]:
    boxes = np.asarray(det.boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(det.scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return []

    h, w = det.image_shape
    image_area = max(float(h * w), 1.0)
    box_area_ratio = area(boxes) / image_area
    proposal_mask = scores >= state_cfg.proposal_min_conf
    difficult_mask = (box_area_ratio <= state_cfg.small_area_ratio) | (scores < env_cfg.high_conf_threshold)
    target_mask = proposal_mask & difficult_mask
    if not target_mask.any():
        target_mask = proposal_mask
    if not target_mask.any():
        return []

    target_boxes = boxes[target_mask]
    target_scores = scores[target_mask]
    target_points = centers(target_boxes)
    target_area_ratio = box_area_ratio[target_mask]

    min_side = min(h, w) * env_cfg.min_slice_fraction
    max_side_by_fraction = min(h, w) * env_cfg.max_slice_fraction
    max_side_by_area = np.sqrt(max(float(h * w) * env_cfg.max_roi_area_ratio, 1.0))
    max_side = max(min(max_side_by_fraction, max_side_by_area), min_side)
    base_side = float(np.clip(min(h, w) * env_cfg.initial_slice_fraction, min_side, max_side))

    object_priority = (1.0 - np.clip(target_scores, 0.0, 1.0)).astype(np.float32)
    object_priority += (target_area_ratio <= state_cfg.small_area_ratio).astype(np.float32) * 0.75

    candidates: list[tuple[float, np.ndarray]] = []
    seen: set[tuple[int, int]] = set()
    for point in target_points:
        key = (int(point[0] // max(base_side * 0.35, 1.0)), int(point[1] // max(base_side * 0.35, 1.0)))
        if key in seen:
            continue
        seen.add(key)
        roi = box_from_center(float(point[0]), float(point[1]), base_side, det.image_shape)
        inside = (
            (target_points[:, 0] >= roi[0])
            & (target_points[:, 0] <= roi[2])
            & (target_points[:, 1] >= roi[1])
            & (target_points[:, 1] <= roi[3])
        )
        if not inside.any():
            continue
        overlap = _max_roi_overlap(roi, previous_rois)
        if overlap >= env_cfg.old_slice_overlap_threshold:
            continue
        score = float(object_priority[inside].sum())
        score += float(inside.sum()) * 0.15
        score -= overlap * 2.0
        candidates.append((score, roi.astype(np.float32)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [roi for _score, roi in candidates[:max_candidates]]


def _max_roi_overlap(roi: np.ndarray, previous_rois: np.ndarray) -> float:
    previous_rois = np.asarray(previous_rois, dtype=np.float32).reshape(-1, 4)
    if len(previous_rois) == 0:
        return 0.0
    roi = np.asarray(roi, dtype=np.float32).reshape(1, 4)
    inter = intersection_matrix(roi, previous_rois)[0]
    current_area = max(float(area(roi)[0]), 1.0)
    return float(np.clip(inter.max() / current_area, 0.0, 1.0))


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
    agnostic_nms_iou: float | None = None,
    slice_score_bonus: float | None = None,
    max_det: int | None = None,
    max_slices: int | None = None,
    device: str | None = None,
    feature_layers: tuple[int, ...] | list[int] | str | None = None,
    min_slice_detections: int | None = None,
    max_slice_attempts: int | None = None,
    class_conf_thresholds: dict | None = None,
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
        agnostic_nms_iou=float(infer_cfg.get("agnostic_nms_iou", 0.0))
        if agnostic_nms_iou is None
        else float(agnostic_nms_iou),
        slice_score_bonus=float(infer_cfg.get("slice_score_bonus", 0.0))
        if slice_score_bonus is None
        else float(slice_score_bonus),
        max_det=_value_or_config(infer_cfg, "max_det", max_det, int),
        max_slices=int(infer_cfg.get("max_slices", 0)) if max_slices is None else int(max_slices),
        device=device if device is not None else project_cfg.optional_str("infer", "device"),
        feature_layers=_feature_layers_or_config(project_cfg, feature_layers),
        min_slice_detections=_value_or_config(infer_cfg, "min_slice_detections", min_slice_detections, int),
        max_slice_attempts=_value_or_config(infer_cfg, "max_slice_attempts", max_slice_attempts, int),
        class_conf_thresholds=infer_cfg.get("class_conf_thresholds", {})
        if class_conf_thresholds is None
        else class_conf_thresholds,
    )
    inferencer = AdaptiveSahiInferencer(weights=weights, checkpoint=checkpoint, cfg=cfg)
    return inferencer.infer_image(
        image_path=image_path,
        out_dir=out_dir,
        cache_root=cache_root,
        split=split,
        use_cache=use_cache,
    )
