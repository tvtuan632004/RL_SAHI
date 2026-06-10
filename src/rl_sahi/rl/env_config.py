from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EnvConfig:
    max_steps: int = 20
    max_slices: int = 8
    initial_slice_fraction: float = 0.35
    move_fraction: float = 0.45
    zoom_factor: float = 0.75
    min_slice_fraction: float = 0.12
    max_slice_fraction: float = 0.45
    max_roi_area_ratio: float = 0.20
    min_scale_gain: float = 2.0
    reward_imgsz: int = 320
    target_projected_size: float = 32.0
    min_projected_size: float = 12.0
    max_projected_size: float = 96.0
    context_margin: float = 0.08
    step_penalty: float = 0.03
    empty_slice_penalty: float = 0.35
    area_penalty: float = 0.35
    detected_overlap_penalty: float = 1.0
    new_hard_reward: float = 2.0
    hard_density_reward: float = 0.8
    stop_target_reward: float = 1.2
    stop_early_penalty: float = 0.8
    high_conf_threshold: float = 0.5
    large_roi_penalty: float = 2.0
    low_scale_penalty: float = 1.0
    old_slice_overlap_threshold: float = 0.5
    old_slice_overlap_penalty: float = 3.0
    min_new_hits_to_accept: int = 1


@dataclass(slots=True)
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict
