from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rl_sahi.common.actions import NUM_ACTIONS, Action
from rl_sahi.common.boxes import as_boxes
from rl_sahi.rl.checkpoint import save_checkpoint
from rl_sahi.rl.dataset import CachedEpisodeDataset
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.network import QNetwork
from rl_sahi.rl.replay import ReplayBuffer
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.state_layout import state_layout_from_detection


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 20000
    batch_size: int = 64
    replay_size: int = 50000
    gamma: float = 0.95
    lr: float = 1e-4
    min_replay: int = 512
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 8000
    guide_prob: float = 0.25
    hidden_dim: int = 512
    use_spatial_cnn: bool = True
    optimize_every: int = 2
    preload_cache: bool = True
    seed: int = 42
    log_interval: int = 25


def epsilon_by_step(step: int, cfg: TrainConfig) -> float:
    frac = min(float(step) / max(cfg.epsilon_decay_steps, 1), 1.0)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def select_action(
    policy: QNetwork,
    state: np.ndarray,
    epsilon: float,
    guide_prob: float,
    env: SliceEnv,
    device: torch.device,
) -> Action:
    if random.random() < epsilon:
        return Action(random.randrange(NUM_ACTIONS))
    if random.random() < guide_prob:
        return env.guided_action()
    with torch.no_grad():
        x = torch.from_numpy(state).float().unsqueeze(0).to(device)
        return Action(int(policy(x).argmax(dim=1).item()))


def optimize(
    policy: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float | None:
    if len(replay) < batch_size:
        return None
    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    states_t = torch.from_numpy(states).float().to(device)
    actions_t = torch.from_numpy(actions).long().to(device)
    rewards_t = torch.from_numpy(rewards).float().to(device)
    next_states_t = torch.from_numpy(next_states).float().to(device)
    dones_t = torch.from_numpy(dones).float().to(device)

    q_values = policy(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(next_states_t).max(dim=1).values
        target_q = rewards_t + gamma * next_q * (1.0 - dones_t)
    loss = F.smooth_l1_loss(q_values, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())


def train_dqn(
    image_root: Path,
    cache_root: Path,
    split: str,
    out_dir: Path,
    cfg: TrainConfig,
    env_cfg: EnvConfig,
    state_cfg: StateConfig,
    limit: int | None = None,
    device_name: str | None = None,
) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    dataset = CachedEpisodeDataset(
        image_root=image_root,
        cache_root=cache_root,
        split=split,
        limit=limit,
        preload=cfg.preload_cache,
    )
    probe_det = dataset.first_detection()
    probe_env = SliceEnv(probe_det, None, env_cfg=env_cfg, state_cfg=state_cfg)
    state_dim = int(probe_env.reset().shape[0])
    layout = state_layout_from_detection(probe_det, state_cfg)
    if layout.state_dim != state_dim:
        raise ValueError(f"State layout mismatch: layout has {layout.state_dim}, env produced {state_dim}")

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    policy = QNetwork(
        state_dim,
        hidden_dim=cfg.hidden_dim,
        layout=layout,
        use_spatial_cnn=cfg.use_spatial_cnn,
    ).to(device)
    target = QNetwork(
        state_dim,
        hidden_dim=cfg.hidden_dim,
        layout=layout,
        use_spatial_cnn=cfg.use_spatial_cnn,
    ).to(device)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    replay = ReplayBuffer(cfg.replay_size)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_reward = -float("inf")
    global_step = 0
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "reward", "loss", "epsilon", "steps", "slices", "covered", "hard_total"],
        )
        writer.writeheader()
        for episode in range(1, cfg.episodes + 1):
            det, hard = dataset.random_episode()
            previous_rois: list[np.ndarray] = []
            previous_covered = np.zeros((len(as_boxes(hard.hard_boxes)),), dtype=bool)
            total_reward = 0.0
            total_steps = 0
            accepted_slices = 0
            losses: list[float] = []
            info = {"covered": 0, "hard_total": len(previous_covered)}

            for _slice_idx in range(env_cfg.max_slices):
                prev_arr = np.stack(previous_rois).astype(np.float32) if previous_rois else np.zeros((0, 4), dtype=np.float32)
                env = SliceEnv(
                    det,
                    hard,
                    env_cfg=env_cfg,
                    state_cfg=state_cfg,
                    previous_rois=prev_arr,
                    previous_covered=previous_covered,
                )
                state = env.reset()

                for _ in range(env_cfg.max_steps + 1):
                    epsilon = epsilon_by_step(global_step, cfg)
                    action = select_action(policy, state, epsilon, cfg.guide_prob, env, device)
                    result = env.step(action)
                    replay.push(state, action, result.reward, result.state, result.done)
                    state = result.state
                    total_reward += result.reward
                    total_steps += 1
                    info = result.info
                    global_step += 1

                    optimize_every = max(int(cfg.optimize_every), 1)
                    if len(replay) >= cfg.min_replay and global_step % optimize_every == 0:
                        loss = optimize(policy, target, optimizer, replay, cfg.batch_size, cfg.gamma, device)
                        if loss is not None:
                            losses.append(loss)
                    if global_step % cfg.target_update == 0:
                        target.load_state_dict(policy.state_dict())
                    if result.done:
                        break

                new_hits = int((env.covered & ~previous_covered).sum())
                if info.get("stop_due_to_old_overlap", False):
                    break
                if new_hits < env_cfg.min_new_hits_to_accept:
                    break
                previous_rois.append(env.roi.copy())
                previous_covered = env.covered.copy()
                accepted_slices += 1
                if previous_covered.all() and len(previous_covered) > 0:
                    break

            mean_loss = float(np.mean(losses)) if losses else 0.0
            row = {
                "episode": episode,
                "reward": round(total_reward, 6),
                "loss": round(mean_loss, 6),
                "epsilon": round(epsilon_by_step(global_step, cfg), 6),
                "steps": total_steps,
                "slices": accepted_slices,
                "covered": int(previous_covered.sum()),
                "hard_total": int(len(previous_covered)),
            }
            writer.writerow(row)
            f.flush()

            if total_reward > best_reward:
                best_reward = total_reward
                save_checkpoint(best_path, policy, state_dim, cfg, env_cfg, state_cfg, layout)
            if episode % cfg.log_interval == 0 or episode == 1:
                print(
                    f"[train] ep={episode}/{cfg.episodes} reward={total_reward:.3f} "
                    f"loss={mean_loss:.4f} eps={epsilon_by_step(global_step, cfg):.3f} "
                    f"slices={accepted_slices} covered={row['covered']}/{row['hard_total']}"
                )

    save_checkpoint(last_path, policy, state_dim, cfg, env_cfg, state_cfg, layout)
    return best_path
