from __future__ import annotations

import numpy as np

from rl_sahi.common.actions import Action
from rl_sahi.common.boxes import (
    area,
    as_boxes,
    box_from_center,
    centers,
    intersection_matrix,
    ioa_matrix,
    rasterize_boxes,
    translate_box,
    zoom_box,
)
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.rl.env_config import EnvConfig, StepResult
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_maps import build_detection_map, mark_history, proposal_mask, proposal_quality
from rl_sahi.rl.state_summary import detection_summary
from rl_sahi.rl.state_vector import build_state_vector


class SliceEnv:
    def __init__(
        self,
        detection: DetectionCache,
        hard_regions: HardRegionCache | None,
        env_cfg: EnvConfig | None = None,
        state_cfg: StateConfig | None = None,
        previous_rois: np.ndarray | None = None,
        previous_covered: np.ndarray | None = None,
    ) -> None:
        self.detection = detection
        self.hard_regions = hard_regions
        self.env_cfg = env_cfg or EnvConfig()
        self.state_cfg = state_cfg or StateConfig()
        self.image_shape = detection.image_shape
        self.detection_map = build_detection_map(detection.boxes, detection.scores, self.image_shape, self.state_cfg)
        self.hard_boxes = as_boxes(hard_regions.hard_boxes if hard_regions is not None else np.zeros((0, 4)))
        self.previous_rois = as_boxes(previous_rois if previous_rois is not None else np.zeros((0, 4), dtype=np.float32))
        self.previous_slice_map = self._build_previous_slice_map()
        self.previous_covered = self._init_previous_covered(previous_covered)
        self.history = np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        self.covered = self.previous_covered.copy()
        self.roi = self._initial_roi()
        self.step_index = 0

    def reset(self) -> np.ndarray:
        self.history.fill(0.0)
        self.covered = self.previous_covered.copy()
        self.roi = self._initial_roi()
        self.step_index = 0
        self.history = mark_history(self.history, self.roi, self.image_shape, self.state_cfg.grid_size)
        return self._state()

    def step(self, action: int | Action) -> StepResult:
        action = Action(int(action))
        done = action == Action.STOP
        if action != Action.STOP:
            self.roi = self._apply_action(action)
            self.step_index += 1
            self.history = mark_history(self.history, self.roi, self.image_shape, self.state_cfg.grid_size)

        reward, info = self._reward(action)
        if info["old_slice_overlap"] >= self.env_cfg.old_slice_overlap_threshold:
            done = True
            info["stop_due_to_old_overlap"] = True
        else:
            info["stop_due_to_old_overlap"] = False
        if self.step_index >= self.env_cfg.max_steps:
            done = True
        info["roi"] = self.roi.copy()
        info["covered"] = int(self.covered.sum())
        info["hard_total"] = int(len(self.hard_boxes))
        return StepResult(self._state(), reward, done, info)

    def guided_action(self) -> Action:
        heatmap_target = self._heatmap_target()
        boxes = as_boxes(self.detection.boxes)
        scores = np.asarray(self.detection.scores, dtype=np.float32).reshape(-1)
        valid_mask = scores >= self.state_cfg.proposal_min_conf
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        if len(boxes) == 0:
            return self._action_toward_target(heatmap_target[0]) if heatmap_target is not None else Action.STOP

        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        det_area_ratio = area(boxes) / image_area
        prop_mask = proposal_mask(scores, self.state_cfg)
        small_mask = det_area_ratio <= self.state_cfg.small_area_ratio
        target_mask = prop_mask | (small_mask & (scores < self.env_cfg.high_conf_threshold))
        if not target_mask.any():
            return self._action_toward_target(heatmap_target[0]) if heatmap_target is not None else Action.STOP

        candidate_boxes = boxes[target_mask]
        candidate_scores = scores[target_mask]
        candidate_centers = centers(candidate_boxes)
        if len(self.previous_rois) > 0:
            old_seen = self._points_in_previous_rois(candidate_centers)
        else:
            old_seen = np.zeros((len(candidate_boxes),), dtype=bool)

        roi_center = centers(self.roi.reshape(1, 4))[0]
        distances = np.linalg.norm(candidate_centers - roi_center[None, :], axis=1)
        quality = proposal_quality(candidate_scores, self.state_cfg)
        heat_support = self._objectness_values_at_points(candidate_centers)
        density_support = self._proposal_density_values_at_points(candidate_centers)
        high_seen = self._points_in_boxes(
            candidate_centers,
            boxes[scores >= self.env_cfg.high_conf_threshold],
        )
        priority = quality
        priority += small_mask[target_mask].astype(np.float32) * 0.5
        priority += heat_support * 0.5
        priority += density_support * 0.75
        priority -= distances / max(min(self.image_shape), 1)
        priority -= old_seen.astype(np.float32) * 2.0
        priority -= high_seen.astype(np.float32) * 1.0
        target_idx = int(priority.argmax())
        if heatmap_target is not None:
            heat_point, heat_score = heatmap_target
            heat_distance = float(np.linalg.norm(heat_point - roi_center) / max(min(self.image_shape), 1))
            heat_priority = float(heat_score - heat_distance)
            if heat_priority > float(priority[target_idx]):
                return self._action_toward_target(heat_point)
        if priority[target_idx] < -1.5:
            return Action.STOP
        return self._action_toward_target(candidate_centers[target_idx], candidate_boxes[[target_idx]])

    def _heatmap_target(self) -> tuple[np.ndarray, float] | None:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return None
        grid_size = self.state_cfg.grid_size
        obj = np.nan_to_num(obj.reshape(-1, grid_size, grid_size), nan=0.0, posinf=0.0, neginf=0.0)
        heat = obj.max(axis=0)
        if self.detection_map.shape[0] > 2:
            density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
            heat = np.maximum(heat * 0.7, density)
        if heat.size == 0:
            return None
        priority = heat.copy()
        priority -= 0.6 * np.asarray(self.previous_slice_map, dtype=np.float32)
        priority -= 0.2 * np.asarray(self.history, dtype=np.float32)
        y, x = np.unravel_index(int(priority.argmax()), priority.shape)
        score = float(priority[y, x])
        if score <= 0.02:
            return None
        h, w = self.image_shape
        target = np.array([(x + 0.5) * w / grid_size, (y + 0.5) * h / grid_size], dtype=np.float32)
        return target, score

    def _objectness_values_at_points(self, points: np.ndarray) -> np.ndarray:
        obj = np.asarray(self.detection.objectness_map, dtype=np.float32)
        if obj.size == 0:
            return np.zeros((len(points),), dtype=np.float32)
        grid = obj.reshape(-1, self.state_cfg.grid_size, self.state_cfg.grid_size).max(axis=0)
        return self._grid_values_at_points(grid, points)

    def _proposal_density_values_at_points(self, points: np.ndarray) -> np.ndarray:
        if self.detection_map.shape[0] <= 2:
            return np.zeros((len(points),), dtype=np.float32)
        density = np.clip(self.detection_map[2] * self.state_cfg.count_norm / 10.0, 0.0, 1.0)
        return self._grid_values_at_points(density, points)

    def _grid_values_at_points(self, grid: np.ndarray, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0:
            return np.zeros((0,), dtype=np.float32)
        grid = np.asarray(grid, dtype=np.float32).reshape(self.state_cfg.grid_size, self.state_cfg.grid_size)
        h, w = self.image_shape
        xs = np.clip((points[:, 0] / max(w, 1)) * self.state_cfg.grid_size, 0, self.state_cfg.grid_size - 1).astype(int)
        ys = np.clip((points[:, 1] / max(h, 1)) * self.state_cfg.grid_size, 0, self.state_cfg.grid_size - 1).astype(int)
        return grid[ys, xs].astype(np.float32)

    def _points_in_boxes(self, points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = as_boxes(boxes)
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if len(points) == 0 or len(boxes) == 0:
            return np.zeros((len(points),), dtype=bool)
        mask = np.zeros((len(points),), dtype=bool)
        for box in boxes:
            mask |= (
                (points[:, 0] >= box[0])
                & (points[:, 0] <= box[2])
                & (points[:, 1] >= box[1])
                & (points[:, 1] <= box[3])
            )
        return mask

    def _action_toward_target(self, target: np.ndarray, target_box: np.ndarray | None = None) -> Action:
        target = np.asarray(target, dtype=np.float32).reshape(2)
        roi_center = centers(self.roi.reshape(1, 4))[0]
        x1, y1, x2, y2 = self.roi
        inside = x1 <= target[0] <= x2 and y1 <= target[1] <= y2
        if inside:
            if self._roi_area_ratio() > self.env_cfg.max_roi_area_ratio or self._scale_gain() < self.env_cfg.min_scale_gain:
                return Action.ZOOM_IN
            if target_box is not None:
                projected_size = self._projected_sizes(target_box)[0]
                if projected_size < self.env_cfg.target_projected_size:
                    return Action.ZOOM_IN
                if projected_size > self.env_cfg.max_projected_size:
                    return Action.ZOOM_OUT
            min_side, _max_side = self._side_limits()
            if self._roi_side() > min_side * 1.25:
                return Action.ZOOM_IN
            return Action.STOP

        dx = target[0] - roi_center[0]
        dy = target[1] - roi_center[1]
        if abs(dx) >= abs(dy):
            return Action.RIGHT if dx > 0 else Action.LEFT
        return Action.DOWN if dy > 0 else Action.UP

    def _initial_roi(self) -> np.ndarray:
        h, w = self.image_shape
        _min_side, max_side = self._side_limits()
        side = min(min(h, w) * self.env_cfg.initial_slice_fraction, max_side)
        return box_from_center(w / 2.0, h / 2.0, side, self.image_shape)

    def _apply_action(self, action: Action) -> np.ndarray:
        side = self._roi_side()
        step = side * self.env_cfg.move_fraction
        if action == Action.LEFT:
            return translate_box(self.roi, -step, 0.0, self.image_shape)
        if action == Action.RIGHT:
            return translate_box(self.roi, step, 0.0, self.image_shape)
        if action == Action.UP:
            return translate_box(self.roi, 0.0, -step, self.image_shape)
        if action == Action.DOWN:
            return translate_box(self.roi, 0.0, step, self.image_shape)
        min_side, max_side = self._side_limits()
        if action == Action.ZOOM_IN:
            return zoom_box(self.roi, self.env_cfg.zoom_factor, self.image_shape, min_side, max_side)
        if action == Action.ZOOM_OUT:
            return zoom_box(self.roi, 1.0 / self.env_cfg.zoom_factor, self.image_shape, min_side, max_side)
        return self.roi

    def _side_limits(self) -> tuple[float, float]:
        h, w = self.image_shape
        min_side = min(h, w) * self.env_cfg.min_slice_fraction
        max_side_by_fraction = min(h, w) * self.env_cfg.max_slice_fraction
        max_side_by_area = np.sqrt(max(float(h * w) * self.env_cfg.max_roi_area_ratio, 1.0))
        max_side = max(min(max_side_by_fraction, max_side_by_area), min_side)
        return float(min_side), float(max_side)

    def _build_previous_slice_map(self) -> np.ndarray:
        if len(self.previous_rois) == 0:
            return np.zeros((self.state_cfg.grid_size, self.state_cfg.grid_size), dtype=np.float32)
        return rasterize_boxes(self.previous_rois, self.image_shape, self.state_cfg.grid_size)

    def _init_previous_covered(self, previous_covered: np.ndarray | None) -> np.ndarray:
        if len(self.hard_boxes) == 0:
            return np.zeros((0,), dtype=bool)
        if previous_covered is not None:
            arr = np.asarray(previous_covered, dtype=bool).reshape(-1)
            if len(arr) != len(self.hard_boxes):
                raise ValueError("previous_covered length must match hard region count")
            return arr.copy()
        covered = np.zeros((len(self.hard_boxes),), dtype=bool)
        for roi in self.previous_rois:
            _scores, hit_mask = self._hard_target_scores(roi)
            covered |= hit_mask
        return covered

    def _points_in_previous_rois(self, points: np.ndarray) -> np.ndarray:
        mask = np.zeros((len(points),), dtype=bool)
        for roi in self.previous_rois:
            mask |= (
                (points[:, 0] >= roi[0])
                & (points[:, 0] <= roi[2])
                & (points[:, 1] >= roi[1])
                & (points[:, 1] <= roi[3])
            )
        return mask

    def _old_slice_overlap(self, roi: np.ndarray | None = None) -> float:
        if len(self.previous_rois) == 0:
            return 0.0
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        inter = intersection_matrix(roi.reshape(1, 4), self.previous_rois)[0]
        current_area = max(float(area(roi.reshape(1, 4))[0]), 1.0)
        return float(np.clip(inter.max() / current_area, 0.0, 1.0))

    def _roi_side(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        return max(float(roi[2] - roi[0]), float(roi[3] - roi[1]), 1.0)

    def _roi_area_ratio(self, roi: np.ndarray | None = None) -> float:
        roi = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        image_area = max(float(self.image_shape[0] * self.image_shape[1]), 1.0)
        return float(area(roi.reshape(1, 4))[0] / image_area)

    def _scale_gain(self, roi: np.ndarray | None = None) -> float:
        return float(min(self.image_shape) / self._roi_side(roi))

    def _projected_sizes(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return np.zeros((0,), dtype=np.float32)
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        return (np.maximum(widths, heights) * float(self.env_cfg.reward_imgsz) / self._roi_side(roi)).astype(np.float32)

    def _projected_size_scores(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        projected = self._projected_sizes(boxes, roi)
        if len(projected) == 0:
            return projected
        cfg = self.env_cfg
        below_target = (projected - cfg.min_projected_size) / max(cfg.target_projected_size - cfg.min_projected_size, 1e-6)
        above_target = (cfg.max_projected_size - projected) / max(cfg.max_projected_size - cfg.target_projected_size, 1e-6)
        return np.clip(np.minimum(below_target, above_target), 0.0, 1.0).astype(np.float32)

    def _center_context_mask(self, boxes: np.ndarray, roi: np.ndarray | None = None) -> np.ndarray:
        boxes = as_boxes(boxes)
        if len(boxes) == 0:
            return np.zeros((0,), dtype=bool)
        x1, y1, x2, y2 = self.roi if roi is None else np.asarray(roi, dtype=np.float32).reshape(4)
        width = max(float(x2 - x1), 1.0)
        height = max(float(y2 - y1), 1.0)
        margin_x = width * self.env_cfg.context_margin
        margin_y = height * self.env_cfg.context_margin
        pts = centers(boxes)
        return (
            (pts[:, 0] >= x1 + margin_x)
            & (pts[:, 0] <= x2 - margin_x)
            & (pts[:, 1] >= y1 + margin_y)
            & (pts[:, 1] <= y2 - margin_y)
        )

    def _hard_target_scores(self, roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if len(self.hard_boxes) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
        if self._roi_area_ratio(roi) > self.env_cfg.max_roi_area_ratio or self._scale_gain(roi) < self.env_cfg.min_scale_gain:
            return np.zeros((len(self.hard_boxes),), dtype=np.float32), np.zeros((len(self.hard_boxes),), dtype=bool)
        context_mask = self._center_context_mask(self.hard_boxes, roi)
        size_scores = self._projected_size_scores(self.hard_boxes, roi)
        target_scores = np.where(context_mask, size_scores, 0.0).astype(np.float32)
        return target_scores, target_scores > 0.0

    def _state(self) -> np.ndarray:
        summary = detection_summary(
            boxes=self.detection.boxes,
            scores=self.detection.scores,
            roi=self.roi,
            history=self.history,
            previous_slice_map=self.previous_slice_map,
            image_shape=self.image_shape,
            step_index=self.step_index,
            max_steps=self.env_cfg.max_steps,
            old_slice_overlap=self._old_slice_overlap(),
            scale_gain=self._scale_gain(),
            previous_slice_count=len(self.previous_rois),
            cfg=self.state_cfg,
        )
        return build_state_vector(
            self.detection.feature,
            self.history,
            self.previous_slice_map,
            self.detection_map,
            self.detection.objectness_map,
            self.detection.spatial_feature_map,
            summary,
        )

    def _update_covered(self) -> tuple[np.ndarray, np.ndarray]:
        target_scores, hit_mask = self._hard_target_scores()
        self.covered |= hit_mask
        return target_scores, hit_mask

    def _reward(self, action: Action) -> tuple[float, dict]:
        prev_covered = self.covered.copy()
        target_scores, hit_mask = self._update_covered()
        new_mask = hit_mask & ~prev_covered
        new_hits = int(new_mask.sum())
        target_score = float(target_scores[new_mask].sum()) if len(target_scores) else 0.0
        hit_count = int(hit_mask.sum()) if len(hit_mask) else 0
        roi_area_ratio = self._roi_area_ratio()
        scale_gain = self._scale_gain()
        old_slice_overlap = self._old_slice_overlap()

        reward = -self.env_cfg.step_penalty
        info = {
            "new_hits": new_hits,
            "hit_count": hit_count,
            "target_score": target_score,
            "roi_area_ratio": roi_area_ratio,
            "scale_gain": scale_gain,
            "old_slice_overlap": old_slice_overlap,
            "detected_overlap": 0.0,
        }

        if len(self.hard_boxes) > 0:
            if new_hits > 0:
                reward += self.env_cfg.new_hard_reward * new_hits
                density = target_score * self.env_cfg.max_roi_area_ratio / max(roi_area_ratio, 1e-6)
                reward += self.env_cfg.hard_density_reward * float(np.clip(density, 0.0, 4.0))
            else:
                reward -= self.env_cfg.empty_slice_penalty
        elif action != Action.STOP:
            reward -= self.env_cfg.empty_slice_penalty

        det_mask = self.detection.scores >= self.env_cfg.high_conf_threshold
        det_boxes = self.detection.boxes[det_mask]
        if len(det_boxes) > 0:
            det_cover = ioa_matrix(self.roi.reshape(1, 4), det_boxes)[0]
            detected_overlap = float(np.clip(det_cover.sum() / max(len(det_boxes), 1), 0.0, 1.0))
            reward -= self.env_cfg.detected_overlap_penalty * detected_overlap
            info["detected_overlap"] = detected_overlap

        reward -= self.env_cfg.area_penalty * roi_area_ratio
        if roi_area_ratio > self.env_cfg.max_roi_area_ratio:
            overflow = roi_area_ratio / max(self.env_cfg.max_roi_area_ratio, 1e-6) - 1.0
            reward -= self.env_cfg.large_roi_penalty * overflow
        if scale_gain < self.env_cfg.min_scale_gain:
            under_scale = self.env_cfg.min_scale_gain / max(scale_gain, 1e-6) - 1.0
            reward -= self.env_cfg.low_scale_penalty * under_scale
        if old_slice_overlap >= self.env_cfg.old_slice_overlap_threshold:
            overflow = old_slice_overlap / max(self.env_cfg.old_slice_overlap_threshold, 1e-6) - 1.0
            reward -= self.env_cfg.old_slice_overlap_penalty * (1.0 + overflow)

        if action == Action.STOP:
            if target_score > 0.0 and old_slice_overlap < self.env_cfg.old_slice_overlap_threshold:
                reward += self.env_cfg.stop_target_reward * min(target_score / max(new_hits, 1), 1.0)
            else:
                reward -= self.env_cfg.stop_early_penalty
        return float(reward), info
