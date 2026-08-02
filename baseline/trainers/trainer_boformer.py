"""Train BOFormer with Q-augmented trajectory replay on DISC environments.

This trainer mirrors the high-level workflow of the DISC training pipeline:
parallel rollout collection, prioritized trajectory replay, and recursive
target-network Q augmentation. The main difference is the state representation:
BOFormer consumes candidate posterior statistics plus incumbent objective values
and therefore pads every task to the largest objective count in the training
set and passes an objective mask alongside the padded tensors.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import trainer as base
from agents.boformer import BOFormer, upgrade_legacy_observation_action_weight
from infill import (
    EPDIExploitation,
    EPDIExploration,
    ExpectedHypervolumeImprovement,
    NDA,
    NDPBIConvergence,
    NDPBIDiversity,
    RandomSelection,
)
from problem.problem import make_problem


TRAJECTORY_BATCH_SIZE = 8
TRAJECTORY_REPLAY_SIZE = 128
PRIORITY_ALPHA = 0.6
PRIORITY_BETA = 0.4
PRIORITY_EPS = 1e-6
TARGET_SYNC_UPDATES = 5
BOFORMER_LR = 1e-5
BOFORMER_WEIGHT_DECAY = 1e-5
BOFORMER_DROPOUT = 0.1
BOFORMER_N_LAYERS = 8
BOFORMER_N_HEADS = 4
BOFORMER_WINDOW_SIZE = 31
BOFORMER_EPSILON = 0.1
BOFORMER_TEMPERATURE = 1000.0
BOFORMER_R_DEMO = 0.01
BOFORMER_REWARD_SCHEME = 2
BOFORMER_GAMMA = 0.99
BOFORMER_UPDATES_PER_EPOCH = 80
DEMO_POLICY_CHOICES = [
    "none",
    "ehvi",
    "random",
    "nd_a",
    "nd_pbi_convergence",
    "nd_pbi_diversity",
    "epdi_exploitation",
    "epdi_exploration",
]


def _configure_parser(parser):
    parser.add_argument(
        "--demo_policy",
        type=str,
        default="ehvi",
        choices=DEMO_POLICY_CHOICES,
    )
    parser.add_argument("--r_demo", type=float, default=BOFORMER_R_DEMO)


def build_demo_infill_criterion(name, ref_point):
    key = str(name).strip().lower().replace("-", "_")
    if key in {"", "none", "off", "disc"}:
        return None
    if key == "ehvi":
        return ExpectedHypervolumeImprovement(ref_point=ref_point, n_samples=128)
    if key == "random":
        return RandomSelection()
    if key == "nd_a":
        return NDA()
    if key == "nd_pbi_convergence":
        return NDPBIConvergence()
    if key == "nd_pbi_diversity":
        return NDPBIDiversity()
    if key == "epdi_exploitation":
        return EPDIExploitation()
    if key == "epdi_exploration":
        return EPDIExploration()
    raise ValueError(f"Unsupported demo_policy: {name}")


def select_demo_action_from_state(state, criterion, seed):
    if criterion is None:
        raise ValueError("A demo infill criterion is required.")
    selected_idx, _ = criterion.select_index(
        archive_y=np.asarray(state["y_true"], dtype=np.float32),
        candidate_mean=np.asarray(state["y_sur"], dtype=np.float32),
        candidate_std=np.asarray(state["sigma_sur"], dtype=np.float32),
        seed=int(seed),
    )
    return int(selected_idx)


class PrioritizedTrajectoryReplayBuffer:
    def __init__(self, capacity: int, alpha: float = PRIORITY_ALPHA):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.trajectories: list[list[dict[str, Any]]] = []
        self.priorities: list[float] = []
        self.position = 0

    def push(self, trajectory: list[dict[str, Any]]) -> None:
        if not trajectory:
            return
        priority = max(self.priorities, default=1.0)
        item = [dict(step) for step in trajectory]
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(item)
            self.priorities.append(priority)
        else:
            self.trajectories[self.position] = item
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = PRIORITY_BETA):
        if not self.trajectories:
            raise ValueError("Cannot sample an empty trajectory replay buffer.")
        count = min(int(batch_size), len(self.trajectories))
        priorities = np.asarray(self.priorities, dtype=np.float64)
        probabilities = np.power(np.maximum(priorities, PRIORITY_EPS), self.alpha)
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.trajectories), size=count, replace=False, p=probabilities)
        weights = np.power(len(self.trajectories) * probabilities[indices], -float(beta))
        weights /= max(float(weights.max()), PRIORITY_EPS)
        trajectories = [self.trajectories[int(idx)] for idx in indices]
        return trajectories, indices.astype(np.int64), weights.astype(np.float32)

    def update_priorities(self, indices, priorities) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = max(float(priority), PRIORITY_EPS)

    def __len__(self) -> int:
        return len(self.trajectories)


def _has_cli_flag(name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in sys.argv[1:])


def _resolve_training_max_objectives(env_specs: list[dict[str, Any]]) -> int:
    max_obj = 0
    for spec in env_specs:
        problem = make_problem(str(spec["problem_name"]), dim=int(spec["dim"]))
        probe = np.full((1, int(spec["dim"])), 0.5, dtype=np.float32)
        n_obj = int(np.asarray(problem.evaluate(probe), dtype=np.float32).shape[1])
        max_obj = max(max_obj, n_obj)
    if max_obj <= 0:
        raise ValueError("Failed to resolve max objective count for BOFormer.")
    return int(max_obj)


def _pad_objective_array(values: np.ndarray, max_objectives: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] > int(max_objectives):
            raise ValueError(f"Objective dimension {arr.shape[0]} exceeds max_objectives={max_objectives}.")
        out = np.zeros(int(max_objectives), dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out
    if arr.ndim == 2:
        if arr.shape[1] > int(max_objectives):
            raise ValueError(f"Objective dimension {arr.shape[1]} exceeds max_objectives={max_objectives}.")
        out = np.zeros((arr.shape[0], int(max_objectives)), dtype=np.float32)
        out[:, : arr.shape[1]] = arr
        return out
    raise ValueError(f"Unsupported objective array rank: {arr.ndim}.")


def _objective_mask(n_objectives: int, max_objectives: int) -> np.ndarray:
    mask = np.zeros(int(max_objectives), dtype=np.float32)
    mask[: int(n_objectives)] = 1.0
    return mask


def _boformer_variance_from_std(values: np.ndarray) -> np.ndarray:
    std_arr = np.asarray(values, dtype=np.float32)
    return np.square(np.maximum(std_arr, 0.0), dtype=np.float32)


def _normalize_boformer_objective_inputs(
    archive_y: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    candidate_mean_arr = np.asarray(candidate_mean, dtype=np.float32)
    candidate_std_arr = np.asarray(candidate_std, dtype=np.float32)
    stacked_y = np.vstack([archive_y_arr, candidate_mean_arr]).astype(np.float32)
    y_min = stacked_y.min(axis=0, keepdims=True)
    y_max = stacked_y.max(axis=0, keepdims=True)
    y_denom = np.clip(y_max - y_min, 1e-12, None)
    archive_y_norm = np.clip((archive_y_arr - y_min) / y_denom, 0.0, 1.0).astype(np.float32)
    candidate_mean_norm = np.clip((candidate_mean_arr - y_min) / y_denom, 0.0, 1.0).astype(np.float32)
    sigma_min = candidate_std_arr.min(axis=0, keepdims=True)
    sigma_max = candidate_std_arr.max(axis=0, keepdims=True)
    sigma_denom = np.clip(sigma_max - sigma_min, 1e-12, None)
    candidate_std_norm = np.clip((candidate_std_arr - sigma_min) / sigma_denom, 0.0, 1.0).astype(np.float32)
    return archive_y_norm, candidate_mean_norm, candidate_std_norm


def _build_boformer_observation(state: dict[str, Any], max_objectives: int) -> dict[str, np.ndarray]:
    archive_y = np.asarray(state["y_true"], dtype=np.float32)
    candidate_mean = np.asarray(state["y_sur"], dtype=np.float32)
    candidate_std = np.asarray(state["sigma_sur"], dtype=np.float32)
    archive_y_arr, candidate_mean_arr, candidate_std_arr = _normalize_boformer_objective_inputs(
        archive_y,
        candidate_mean,
        candidate_std,
    )
    candidate_variance = _boformer_variance_from_std(candidate_std_arr)
    n_candidates, n_obj = candidate_mean.shape
    if archive_y.shape[1] != n_obj or candidate_variance.shape != candidate_mean.shape:
        raise ValueError("Inconsistent objective dimensions in BOFormer observation.")
    incumbent = archive_y_arr.min(axis=0).astype(np.float32)
    return {
        "candidate_mean": _pad_objective_array(candidate_mean_arr, max_objectives),
        "candidate_variance": _pad_objective_array(candidate_variance, max_objectives),
        "best_objectives": _pad_objective_array(incumbent, max_objectives),
        "progress": np.asarray([float(state["progress"][0])], dtype=np.float32),
        "candidate_mask": np.ones(int(n_candidates), dtype=bool),
        "objective_mask": _objective_mask(int(n_obj), int(max_objectives)),
        "n_obj": int(n_obj),
    }


def _forward_obs_kwargs(obs: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    return {
        "candidate_mean": base.to_tensor(obs["candidate_mean"][None, ...], device),
        "candidate_variance": base.to_tensor(obs["candidate_variance"][None, ...], device),
        "best_objectives": base.to_tensor(obs["best_objectives"][None, ...], device),
        "progress": base.to_tensor(obs["progress"].reshape(1, 1), device),
        "objective_mask": torch.as_tensor(obs["objective_mask"], dtype=torch.float32, device=device).unsqueeze(0),
        "candidate_mask": torch.as_tensor(obs["candidate_mask"], dtype=torch.bool, device=device).unsqueeze(0),
    }


def _history_kwargs(
    history_mean: list[np.ndarray],
    history_variance: list[np.ndarray],
    history_best: list[np.ndarray],
    history_progress: list[float],
    history_rewards: list[float],
    history_q_values: list[float],
    history_objective_mask: list[np.ndarray],
    device: str,
) -> dict[str, Any]:
    if not history_mean:
        return {}
    return {
        "history_mean": base.to_tensor(np.stack(history_mean, axis=0)[None, ...], device),
        "history_variance": base.to_tensor(np.stack(history_variance, axis=0)[None, ...], device),
        "history_best_objectives": base.to_tensor(np.stack(history_best, axis=0)[None, ...], device),
        "history_progress": base.to_tensor(np.asarray(history_progress, dtype=np.float32).reshape(1, -1), device),
        "history_rewards": base.to_tensor(np.asarray(history_rewards, dtype=np.float32).reshape(1, -1), device),
        "history_q_values": base.to_tensor(np.asarray(history_q_values, dtype=np.float32).reshape(1, -1), device),
        "history_objective_mask": base.to_tensor(
            np.stack(history_objective_mask, axis=0)[None, ...],
            device,
        ),
        "history_mask": torch.ones(1, len(history_mean), dtype=torch.bool, device=device),
    }


def _trajectory_storage_stats(trajectory: list[dict[str, Any]]) -> dict[str, float]:
    def _entry_bytes(step: dict[str, Any]) -> int:
        total = 0
        for value in step.values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
        return total

    transition_bytes = [_entry_bytes(step) for step in trajectory]
    candidate_rows = [int(step["obs"]["candidate_mean"].shape[0]) for step in trajectory]
    return {
        "payload_bytes": int(sum(transition_bytes)),
        "transition_bytes_mean": float(np.mean(transition_bytes)) if transition_bytes else 0.0,
        "transition_bytes_max": int(max(transition_bytes, default=0)),
        "candidate_rows_mean": float(np.mean(candidate_rows)) if candidate_rows else 0.0,
        "candidate_rows_max": int(max(candidate_rows, default=0)),
    }


def _state_input_nbytes(obs: dict[str, np.ndarray]) -> int:
    return int(
        obs["candidate_mean"].nbytes
        + obs["candidate_variance"].nbytes
        + obs["best_objectives"].nbytes
        + obs["progress"].nbytes
        + obs["objective_mask"].nbytes
        + obs["candidate_mask"].nbytes
    )


def _obs_tensor_kwargs(obs: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    return {
        "candidate_mean": torch.as_tensor(
            obs["candidate_mean"][None, ...],
            dtype=torch.float32,
            device=device,
        ),
        "candidate_variance": torch.as_tensor(
            obs["candidate_variance"][None, ...],
            dtype=torch.float32,
            device=device,
        ),
        "best_objectives": torch.as_tensor(
            obs["best_objectives"][None, ...],
            dtype=torch.float32,
            device=device,
        ),
        "progress": torch.as_tensor(
            obs["progress"].reshape(1, 1),
            dtype=torch.float32,
            device=device,
        ),
        "objective_mask": torch.as_tensor(
            obs["objective_mask"][None, ...],
            dtype=torch.float32,
            device=device,
        ),
        "candidate_mask": torch.as_tensor(
            obs["candidate_mask"][None, ...],
            dtype=torch.bool,
            device=device,
        ),
    }


def _trajectory_update_cache(trajectory: list[dict[str, Any]], device: str) -> dict[str, Any]:
    n_steps = len(trajectory)
    if n_steps <= 0:
        return {}
    selected_mean = np.stack([step["selected_mean"] for step in trajectory], axis=0).astype(np.float32)
    selected_variance = np.stack([step["selected_variance"] for step in trajectory], axis=0).astype(np.float32)
    selected_best = np.stack([step["selected_best"] for step in trajectory], axis=0).astype(np.float32)
    selected_objective_mask = np.stack(
        [step["selected_objective_mask"] for step in trajectory],
        axis=0,
    ).astype(np.float32)
    return {
        "current_kwargs": [_obs_tensor_kwargs(step["obs"], device) for step in trajectory],
        "next_kwargs": [_obs_tensor_kwargs(step["next_obs"], device) for step in trajectory],
        "history_mean": torch.as_tensor(selected_mean, dtype=torch.float32, device=device),
        "history_variance": torch.as_tensor(selected_variance, dtype=torch.float32, device=device),
        "history_best_objectives": torch.as_tensor(selected_best, dtype=torch.float32, device=device),
        "history_progress": torch.as_tensor(
            np.asarray([step["selected_progress"] for step in trajectory], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        "history_rewards": torch.as_tensor(
            np.asarray([step["reward"] for step in trajectory], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        ),
        "history_q_values": torch.empty(n_steps, dtype=torch.float32, device=device),
        "history_objective_mask": torch.as_tensor(
            selected_objective_mask,
            dtype=torch.float32,
            device=device,
        ),
        "history_mask": torch.ones(1, n_steps, dtype=torch.bool, device=device),
    }


def _cached_history_kwargs(cache: dict[str, Any], n_steps: int) -> dict[str, Any]:
    if n_steps <= 0:
        return {}
    return {
        "history_mean": cache["history_mean"][:n_steps].unsqueeze(0),
        "history_variance": cache["history_variance"][:n_steps].unsqueeze(0),
        "history_best_objectives": cache["history_best_objectives"][:n_steps].unsqueeze(0),
        "history_progress": cache["history_progress"][:n_steps].unsqueeze(0),
        "history_rewards": cache["history_rewards"][:n_steps].unsqueeze(0),
        "history_q_values": cache["history_q_values"][:n_steps].unsqueeze(0),
        "history_objective_mask": cache["history_objective_mask"][:n_steps].unsqueeze(0),
        "history_mask": cache["history_mask"][:, :n_steps],
    }


def rollout_boformer_episode(
    policy_state_dict_cpu,
    target_state_dict_cpu,
    cfg_dict,
    problem_name,
    dim,
    seed,
    epsilon,
    r_demo=0.01,
):
    started_at = time.perf_counter()
    device = str(cfg_dict.get("rollout_device", "cpu"))
    if base._debug_enabled(cfg_dict):
        base.dbg(
            f"worker entered | problem={problem_name} | dim={int(dim)} | "
            f"seed={int(seed)} | device={device} | agent=boformer"
        )
        base.dbg("worker init: before seed_everything")
    base.seed_everything(int(seed), include_cuda=str(device).startswith("cuda"))
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after seed_everything")
    max_objectives = int(cfg_dict["boformer_max_objectives"])
    if base._debug_enabled(cfg_dict):
        base.dbg(f"worker init: resolved max_objectives={max_objectives}")
    init_kwargs = {
        "n_objectives": max_objectives,
        "hidden_dim": int(cfg_dict.get("hidden_dim", 128)),
        "n_layers": int(cfg_dict.get("boformer_n_layers", BOFORMER_N_LAYERS)),
        "n_heads": int(cfg_dict.get("boformer_n_heads", BOFORMER_N_HEADS)),
        "window_size": int(cfg_dict.get("boformer_window_size", BOFORMER_WINDOW_SIZE)),
        "dropout": float(cfg_dict.get("boformer_dropout", BOFORMER_DROPOUT)),
        "epsilon": float(epsilon),
        "temperature": float(cfg_dict.get("boformer_temperature", BOFORMER_TEMPERATURE)),
    }
    if base._debug_enabled(cfg_dict):
        base.dbg(
            "worker init: built init_kwargs | "
            f"hidden_dim={init_kwargs['hidden_dim']} | "
            f"layers={init_kwargs['n_layers']} | heads={init_kwargs['n_heads']} | "
            f"window={init_kwargs['window_size']}"
        )
        base.dbg("worker init: before policy instantiate")
    policy = BOFormer(**init_kwargs)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after policy instantiate")
        base.dbg("worker init: before target instantiate")
    target = BOFormer(**init_kwargs)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after target instantiate")
        base.dbg(f"worker init: before policy.to({device})")
    policy = policy.to(device)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after policy.to(device)")
        base.dbg(f"worker init: before target.to({device})")
    target = target.to(device)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after target.to(device)")
        base.dbg("worker init: before checkpoint upgrade")
    policy_state_dict_cpu = upgrade_legacy_observation_action_weight(policy_state_dict_cpu)
    target_state_dict_cpu = upgrade_legacy_observation_action_weight(target_state_dict_cpu)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after checkpoint upgrade")
        base.dbg("worker init: before policy.load_state_dict")
    policy.load_state_dict(policy_state_dict_cpu, strict=True)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after policy.load_state_dict")
        base.dbg("worker init: before target.load_state_dict")
    target.load_state_dict(target_state_dict_cpu, strict=True)
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after target.load_state_dict")
        base.dbg("worker init: before eval()")
    policy.eval()
    target.eval()
    if base._debug_enabled(cfg_dict):
        base.dbg("worker init: after eval()")

    if base._debug_enabled(cfg_dict):
        base.dbg("before env init")
    env = base.DiscSAEAEnv(problem_name, int(dim), int(seed), cfg_dict)
    if base._debug_enabled(cfg_dict):
        base.dbg("after env init")
    state = env.reset()
    demo_name = str(cfg_dict.get("demo_policy", "ehvi")).strip().lower().replace("-", "_")
    demo_criterion = build_demo_infill_criterion(demo_name, env.ref_point)
    if demo_criterion is not None and hasattr(demo_criterion, "reset"):
        demo_criterion.reset()
    use_demo = demo_criterion is not None and float(
        np.random.default_rng(int(seed) + 7919).random()
    ) < float(r_demo)

    trajectory: list[dict[str, Any]] = []
    total_reward = 0.0
    history_mean: list[np.ndarray] = []
    history_variance: list[np.ndarray] = []
    history_best: list[np.ndarray] = []
    history_progress: list[float] = []
    history_rewards: list[float] = []
    history_q_values: list[float] = []
    history_objective_mask: list[np.ndarray] = []
    done = False
    while not done:
        if base._debug_enabled(cfg_dict):
            base.dbg(f"before env step | rollout_step={len(trajectory)}")
        obs = _build_boformer_observation(state, max_objectives=max_objectives)
        common_kwargs = _forward_obs_kwargs(obs, device)
        hist_kwargs = _history_kwargs(
            history_mean,
            history_variance,
            history_best,
            history_progress,
            history_rewards,
            history_q_values,
            history_objective_mask,
            device,
        )
        with torch.no_grad():
            target_out = target(
                **common_kwargs,
                **hist_kwargs,
                decode_type="q_greedy",
            )
            if use_demo:
                action = select_demo_action_from_state(
                    state,
                    demo_criterion,
                    int(seed) + len(trajectory),
                )
            else:
                policy_out = policy(
                    **common_kwargs,
                    **hist_kwargs,
                    decode_type=str(cfg_dict.get("policy_mode", "softmax_sample")),
                    epsilon=float(epsilon),
                )
                action = int(policy_out["action"].reshape(-1)[0].item())
        if base._debug_enabled(cfg_dict):
            base.dbg(f"before agent/select action | action={int(action)}")
        selected_q = float(target_out["q_values"][0, int(action)].detach().cpu())
        next_state, reward, done = env.step(action, state)
        if base._debug_enabled(cfg_dict):
            base.dbg(f"after agent/select action | action={int(action)}")
        next_obs = _build_boformer_observation(next_state, max_objectives=max_objectives)
        trajectory.append(
            {
                "obs": obs,
                "action": int(action),
                "reward": float(reward),
                "done": bool(done),
                "next_obs": next_obs,
                "selected_mean": np.asarray(obs["candidate_mean"][int(action)], dtype=np.float32),
                "selected_variance": np.asarray(obs["candidate_variance"][int(action)], dtype=np.float32),
                "selected_best": np.asarray(obs["best_objectives"], dtype=np.float32),
                "selected_progress": float(obs["progress"][0]),
                "selected_q": float(selected_q),
                "selected_objective_mask": np.asarray(obs["objective_mask"], dtype=np.float32),
            }
        )
        total_reward += float(reward)
        history_mean.append(np.asarray(obs["candidate_mean"][int(action)], dtype=np.float32))
        history_variance.append(np.asarray(obs["candidate_variance"][int(action)], dtype=np.float32))
        history_best.append(np.asarray(obs["best_objectives"], dtype=np.float32))
        history_progress.append(float(obs["progress"][0]))
        history_rewards.append(float(reward))
        history_q_values.append(float(selected_q))
        history_objective_mask.append(np.asarray(obs["objective_mask"], dtype=np.float32))
        state = next_state
        if base._debug_enabled(cfg_dict):
            base.dbg(f"after env step | rollout_step={len(trajectory)} | done={int(done)}")

    finished_at = time.perf_counter()
    return {
        "trajectory": trajectory,
        "episode_reward": float(total_reward),
        "episode_steps": len(trajectory),
        "env_key": base.env_key(problem_name, dim),
        "init_hv": float(env.init_hv),
        "final_hv": float(env.current_hv()),
        "used_demo_episode": int(use_demo),
        "worker_pid": os.getpid(),
        "task_started_at": started_at,
        "task_finished_at": finished_at,
        "task_duration_sec": finished_at - started_at,
    }


def _trajectory_td_backward(policy, target, trajectory, cfg, importance_weight, batch_count):
    cache = _trajectory_update_cache(trajectory, cfg.device)
    td_errors = []
    q_values_seen = []
    targets_seen = []
    target_current = None

    for step_idx, step in enumerate(trajectory):
        action = int(step["action"])
        reward = float(step["reward"])
        done = float(step["done"])
        history_kwargs = _cached_history_kwargs(cache, step_idx)
        current_kwargs = cache["current_kwargs"][step_idx]

        with torch.no_grad():
            if target_current is None:
                target_current = target(
                    **current_kwargs,
                    **history_kwargs,
                    decode_type="q_greedy",
                )
            selected_target_q = float(target_current["q_values"][0, action].detach().cpu())
            cache["history_q_values"][step_idx] = selected_target_q

        policy_out = policy(
            **current_kwargs,
            **history_kwargs,
            decode_type="q_greedy",
        )
        q_sa = policy_out["q_values"][0, action]
        next_history_kwargs = _cached_history_kwargs(cache, step_idx + 1)

        with torch.no_grad():
            td_target = torch.as_tensor(reward, dtype=q_sa.dtype, device=cfg.device)
            target_next = None
            if not done:
                target_next = target(
                    **cache["next_kwargs"][step_idx],
                    **next_history_kwargs,
                    decode_type="q_greedy",
                )
                next_q = target_next["q_values"].max(dim=1).values[0]
                td_target = td_target + float(cfg.gamma) * next_q

        td_error = q_sa - td_target
        step_loss = td_error.square()
        (step_loss * float(importance_weight) / float(batch_count)).backward()
        td_errors.append(float(td_error.detach().cpu()))
        q_values_seen.append(float(q_sa.detach().cpu()))
        targets_seen.append(float(td_target.detach().cpu()))

        target_current = target_next

    squared_error_sum = float(np.square(np.asarray(td_errors, dtype=np.float64)).sum())
    return {
        "priority": squared_error_sum + PRIORITY_EPS,
        "td_loss": squared_error_sum / max(len(td_errors), 1),
        "q_mean": float(np.mean(q_values_seen)) if q_values_seen else 0.0,
        "target_mean": float(np.mean(targets_seen)) if targets_seen else 0.0,
        "td_error_mean": float(np.mean(td_errors)) if td_errors else 0.0,
        "steps": len(td_errors),
    }


def update_from_trajectory_batch(policy, target, optimizer, replay, cfg):
    sample_started_at = time.perf_counter()
    trajectories, indices, weights = replay.sample(TRAJECTORY_BATCH_SIZE, PRIORITY_BETA)
    sample_cpu_sec = time.perf_counter() - sample_started_at
    if bool(getattr(cfg, "debug", False)):
        storage_stats = [_trajectory_storage_stats(item) for item in trajectories]
        state_input_bytes = [
            _state_input_nbytes(step["obs"]) + _state_input_nbytes(step["next_obs"])
            for trajectory in trajectories
            for step in trajectory
        ]
        trajectory_batch_cpu_mb = float(
            sum(item["payload_bytes"] for item in storage_stats) / (1024.0 * 1024.0)
        )
        state_input_mb_max = float(max(state_input_bytes, default=0) / (1024.0 * 1024.0))
        candidate_rows_mean = float(np.mean([item["candidate_rows_mean"] for item in storage_stats]))
        candidate_rows_max = int(max((item["candidate_rows_max"] for item in storage_stats), default=0))
    else:
        trajectory_batch_cpu_mb = 0.0
        state_input_mb_max = 0.0
        candidate_rows_mean = 0.0
        candidate_rows_max = 0
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    for trajectory, weight in zip(trajectories, weights):
        metrics.append(
            _trajectory_td_backward(
                policy,
                target,
                trajectory,
                cfg,
                importance_weight=float(weight),
                batch_count=len(trajectories),
            )
        )
    grad_norm = float(nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0))
    optimizer.step()
    replay.update_priorities(indices, [item["priority"] for item in metrics])
    return {
        "td_loss": float(np.mean([item["td_loss"] for item in metrics])),
        "q_mean": float(np.mean([item["q_mean"] for item in metrics])),
        "target_mean": float(np.mean([item["target_mean"] for item in metrics])),
        "td_error_mean": float(np.mean([item["td_error_mean"] for item in metrics])),
        "grad_norm": grad_norm,
        "trajectory_steps": float(np.mean([item["steps"] for item in metrics])),
        "sample_cpu_sec": float(sample_cpu_sec),
        "trajectory_batch_cpu_mb": trajectory_batch_cpu_mb,
        "state_input_mb_max": state_input_mb_max,
        "candidate_rows_mean": candidate_rows_mean,
        "candidate_rows_max": candidate_rows_max,
    }


def _build_config(args):
    cfg = base.TrainConfig()
    cfg.seed = int(args.seed)
    cfg.agent_name = "boformer"
    cfg.agent_log_key = "boformer"
    cfg.heldout_problem = str(args.problem).upper()
    cfg.init_size = 80
    cfg.max_fe = 120
    cfg.train_iters = cfg.train_iters if args.epoch is None else int(args.epoch)
    cfg.gamma = BOFORMER_GAMMA if args.gamma is None else float(args.gamma)
    cfg.reward_scheme = BOFORMER_REWARD_SCHEME
    cfg.reward_lambda = float(base.resolve_default_reward_lambda(cfg.reward_scheme))
    # BOFormer uses raw reward-scheme-2 targets without per-trajectory normalization.
    cfg.reward_norm = False
    cfg.surrogate_model = "gp" if str(args.solver).lower() in {"moead_ego", "hybrid"} else str(args.surrogate_model).lower()
    cfg.solver = str(args.solver).lower()
    cfg.saea_steps = 15
    cfg.eval_batch_size = int(args.batch)
    cfg.offspring_size = int(base.default_training_offspring_size_for_solver(cfg.solver))
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    cfg.updates_per_epoch = (
        BOFORMER_UPDATES_PER_EPOCH
        if args.updates_per_epoch is None
        else int(args.updates_per_epoch)
    )
    cfg.hidden_dim = 128 if args.hidden_dim is None else int(args.hidden_dim)
    if args.device is not None:
        cfg.device = str(args.device)
    cfg.rollout_device = str(args.rollout_device)
    cfg.surrogate_device = str(args.surrogate_device)
    cfg.demo_policy = str(args.demo_policy).strip().lower().replace("-", "_")
    cfg.r_demo = float(args.r_demo)
    cfg.debug = False
    cfg.cuda_cleanup_before_update = (
        True if args.cuda_cleanup_before_update is None else bool(args.cuda_cleanup_before_update)
    )
    cfg.cuda_cleanup_after_update = (
        True if args.cuda_cleanup_after_update is None else bool(args.cuda_cleanup_after_update)
    )
    cfg.lr = BOFORMER_LR
    cfg.dropout = BOFORMER_DROPOUT
    cfg.epsilon_start = BOFORMER_EPSILON
    cfg.epsilon_end = BOFORMER_EPSILON
    cfg.epsilon_decay_iters = 1
    cfg.policy_mode = "softmax_sample"
    cfg.boformer_weight_decay = BOFORMER_WEIGHT_DECAY
    cfg.boformer_n_layers = BOFORMER_N_LAYERS
    cfg.boformer_n_heads = BOFORMER_N_HEADS
    cfg.boformer_window_size = BOFORMER_WINDOW_SIZE
    cfg.boformer_dropout = BOFORMER_DROPOUT
    cfg.boformer_temperature = BOFORMER_TEMPERATURE
    return cfg


def train_boformer(args):
    cfg = _build_config(args)
    base.seed_everything(cfg.seed, include_cuda=True)
    env_specs = base.build_training_env_specs(cfg.heldout_problem)
    workers = min(int(cfg.num_workers), len(env_specs))
    if cfg.demo_policy not in DEMO_POLICY_CHOICES:
        raise ValueError(f"Unsupported demo_policy: {cfg.demo_policy}.")
    if not 0.0 <= cfg.r_demo <= 1.0:
        raise ValueError(f"r_demo must be in [0, 1], got {cfg.r_demo}.")
    cfg.boformer_max_objectives = _resolve_training_max_objectives(env_specs)

    log_root = os.path.join("training_logs", "boformer")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"boformer_trainer_{cfg.heldout_problem.lower()}_set1_{timestamp}.txt"
    run_dir, log_path = base._prepare_training_run_layout(log_root, filename)
    cfg.weight_dir = run_dir
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(message):
        print(message)
        log_file.write(str(message) + "\n")

    init_kwargs = {
        "n_objectives": int(cfg.boformer_max_objectives),
        "hidden_dim": int(cfg.hidden_dim),
        "n_layers": int(cfg.boformer_n_layers),
        "n_heads": int(cfg.boformer_n_heads),
        "window_size": int(cfg.boformer_window_size),
        "dropout": float(cfg.boformer_dropout),
        "epsilon": float(cfg.epsilon_start),
        "temperature": float(cfg.boformer_temperature),
    }
    agent = BOFormer(**init_kwargs).to(cfg.device)
    target = copy.deepcopy(agent).to(cfg.device).eval()
    optimizer = optim.Adam(
        agent.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.boformer_weight_decay),
    )
    replay = PrioritizedTrajectoryReplayBuffer(TRAJECTORY_REPLAY_SIZE)
    best_reward = -float("inf")

    cfg_dict = cfg.__dict__.copy()
    log(
        "Training config | "
        f"seed={cfg.seed} | "
        f"heldout={cfg.heldout_problem} | training_set=1 | "
        f"envs={len(env_specs)} | workers={workers} | reward_scheme={cfg.reward_scheme} | "
        f"reward_lambda={cfg.reward_lambda:.4f} | reward_norm={int(cfg.reward_norm)} | "
        f"agent=boformer | learning=non_markovian_q | "
        f"demo_policy={cfg.demo_policy} | r_demo={cfg.r_demo:.4f} | solver={cfg.solver} | "
        f"surrogate={base.effective_surrogate_label(cfg)} | epochs={cfg.train_iters} | "
        f"saea_steps={cfg.saea_steps} | eval_batch_size={cfg.eval_batch_size} | "
        f"updates_per_epoch={cfg.updates_per_epoch} | hidden_dim={cfg.hidden_dim} | "
        f"max_objectives={cfg.boformer_max_objectives} | "
        f"gpt_layers={cfg.boformer_n_layers} | gpt_heads={cfg.boformer_n_heads} | "
        f"window_size={cfg.boformer_window_size} | dropout={cfg.boformer_dropout:.3f} | "
        f"softmax_temperature={cfg.boformer_temperature:.1f} | "
        f"debug={int(cfg.debug)} | "
        f"cuda_cleanup_before_update={int(cfg.cuda_cleanup_before_update)} | "
        f"cuda_cleanup_after_update={int(cfg.cuda_cleanup_after_update)} | "
        f"trajectory_batch_size={TRAJECTORY_BATCH_SIZE} | trajectory_replay_size={TRAJECTORY_REPLAY_SIZE} | "
        f"priority_alpha={PRIORITY_ALPHA:.2f} | priority_beta={PRIORITY_BETA:.2f} | "
        f"target_sync_updates={TARGET_SYNC_UPDATES} | train_device={cfg.device} | "
        f"rollout_device={cfg.rollout_device} | lr={cfg.lr:.1e} | "
        f"weight_decay={cfg.boformer_weight_decay:.1e} | gamma={cfg.gamma:.4f} | "
        f"log_path={log_path}"
    )
    executor = None
    ray_module = None
    ray_worker = None
    if bool(args.ray):
        try:
            import ray as ray_module  # type: ignore
        except ImportError as exc:
            raise ImportError("ray is not available. Install ray or run without --ray.") from exc
        if not ray_module.is_initialized():
            ray_module.init(num_cpus=workers, ignore_reinit_error=True)
        ray_worker = ray_module.remote(num_cpus=1)(rollout_boformer_episode)
    total_update_steps = 0
    last_target_sync_update = 0
    try:
        for local_epoch in range(cfg.train_iters):
            epoch = local_epoch + 1
            epsilon = base.epsilon_by_iter(local_epoch, cfg)
            log(
                f"[Epoch {epoch:04d}] start | solver={cfg.solver} | "
                f"demo_policy={cfg.demo_policy} | r_demo={cfg.r_demo:.3f} | eps={epsilon:.3f}"
            )
            if base._debug_enabled(cfg):
                base.dbg(f"epoch {epoch} after start log")
                base.dbg(f"num_workers={workers}")
                base.dbg("before rollout collection")
            policy_cpu = base.clone_state_dict_cpu(agent)
            target_cpu = base.clone_state_dict_cpu(target)
            if ray_worker is None:
                if base._debug_enabled(cfg):
                    base.dbg(f"before ProcessPoolExecutor create | workers={workers}")
                executor = ProcessPoolExecutor(max_workers=workers)
                if base._debug_enabled(cfg):
                    base.dbg("after ProcessPoolExecutor create")
            futures = []
            for env_idx, spec in enumerate(env_specs):
                seed = int(cfg.seed) + 100000 * epoch + 1000 * env_idx
                rollout_args = (
                    policy_cpu,
                    target_cpu,
                    cfg_dict,
                    spec["problem_name"],
                    int(spec["dim"]),
                    int(seed),
                    float(epsilon),
                    float(cfg.r_demo),
                )
                if ray_worker is not None:
                    if base._debug_enabled(cfg):
                        base.dbg(f"before submit worker {len(futures)}")
                    futures.append(ray_worker.remote(*rollout_args))
                    if base._debug_enabled(cfg):
                        base.dbg(f"after submit worker {len(futures) - 1}")
                else:
                    if base._debug_enabled(cfg):
                        base.dbg(f"before submit worker {len(futures)}")
                    futures.append(executor.submit(rollout_boformer_episode, *rollout_args))
                    if base._debug_enabled(cfg):
                        base.dbg(f"after submit worker {len(futures) - 1}")
            if ray_worker is not None:
                if base._debug_enabled(cfg):
                    base.dbg("before rollout collection ray.get")
                results = ray_module.get(futures)
                if base._debug_enabled(cfg):
                    base.dbg("after rollout collection ray.get")
            else:
                results = []
                for idx, future in enumerate(futures):
                    if base._debug_enabled(cfg):
                        base.dbg(f"before future.result worker {idx}")
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        if base._debug_enabled(cfg):
                            base.dbg(f"future {idx} raised: {repr(exc)}")
                        raise
                    if base._debug_enabled(cfg):
                        base.dbg(f"after future.result worker {idx}")
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None
            if base._debug_enabled(cfg):
                base.dbg("after rollout collection")
            parallel_stats = base.summarize_parallel_rollout(results)
            demo_episodes = sum(int(item["used_demo_episode"]) for item in results)
            if parallel_stats is not None:
                log(
                    f"[Epoch {epoch:04d}] interact parallel | "
                    f"tasks={parallel_stats['tasks']} | "
                    f"unique_pids={parallel_stats['unique_pids']} | "
                    f"pid_list={parallel_stats['pid_list']} | "
                    f"overlap_wall_sec={parallel_stats['overlap_wall_sec']:.3f} | "
                    f"sum_task_sec={parallel_stats['sum_task_sec']:.3f} | "
                    f"mean_task_sec={parallel_stats['mean_task_sec']:.3f} | "
                    f"min_task_sec={parallel_stats['min_task_sec']:.3f} | "
                    f"max_task_sec={parallel_stats['max_task_sec']:.3f} | "
                    f"parallelism_est={parallel_stats['parallelism_est']:.2f} | "
                    f"demo_episodes={demo_episodes} | "
                    f"policy_episodes={len(results) - demo_episodes}"
                )
            for result in results:
                replay.push(result["trajectory"])

            env_stats = {}
            for result in results:
                bucket = env_stats.setdefault(
                    result["env_key"],
                    {"reward": [], "policy_reward": [], "init": [], "final": []},
                )
                steps = max(int(result["episode_steps"]), 1)
                reward_per_fe = float(result["episode_reward"]) / steps
                bucket["reward"].append(reward_per_fe)
                if not int(result["used_demo_episode"]):
                    bucket["policy_reward"].append(reward_per_fe)
                bucket["init"].append(float(result["init_hv"]))
                bucket["final"].append(float(result["final_hv"]))
            demo_enabled = cfg.demo_policy != "none" and cfg.r_demo > 0.0
            policy_rewards = [value for bucket in env_stats.values() for value in bucket["policy_reward"]]
            all_rewards = [value for bucket in env_stats.values() for value in bucket["reward"]]
            mean_reward = (
                float(np.mean(policy_rewards))
                if demo_enabled and policy_rewards
                else (0.0 if demo_enabled else float(np.mean(all_rewards)))
            )
            for key, bucket in sorted(env_stats.items()):
                log(
                    f"{key} epoch {epoch} done, mean reward/FE = {np.mean(bucket['reward']):.4f}, "
                    f"policy mean reward/FE = "
                    f"{np.mean(bucket['policy_reward']) if bucket['policy_reward'] else 0.0:.4f}, "
                    f"init HV = {np.mean(bucket['init']):.6f}, final HV = {np.mean(bucket['final']):.6f}"
                )
            log(
                f"[Epoch {epoch:04d}] rollout done | trajectories={len(results)} | "
                f"demo_episodes={demo_episodes} | policy_episodes={len(results) - demo_episodes} | "
                f"replay={len(replay)} | mean reward/FE={mean_reward:.4f}"
            )
            if cfg.debug:
                new_stats = [_trajectory_storage_stats(item["trajectory"]) for item in results]
                replay_stats = [_trajectory_storage_stats(item) for item in replay.trajectories]
                log(
                    f"[Epoch {epoch:04d}] rollout debug | "
                    f"new_payload_mb={sum(item['payload_bytes'] for item in new_stats) / (1024.0 * 1024.0):.2f} | "
                    f"replay_payload_mb={sum(item['payload_bytes'] for item in replay_stats) / (1024.0 * 1024.0):.2f} | "
                    f"trajectory_mb_mean={np.mean([item['payload_bytes'] for item in new_stats]) / (1024.0 * 1024.0):.3f} | "
                    f"transition_mb_mean={np.mean([item['transition_bytes_mean'] for item in new_stats]) / (1024.0 * 1024.0):.3f} | "
                    f"transition_mb_max={max((item['transition_bytes_max'] for item in new_stats), default=0) / (1024.0 * 1024.0):.3f} | "
                    f"candidate_rows_mean={np.mean([item['candidate_rows_mean'] for item in new_stats]):.2f} | "
                    f"candidate_rows_max={max((item['candidate_rows_max'] for item in new_stats), default=0)}"
                )
            log(
                f"epoch {epoch} pre-update | mean reward/FE = {mean_reward:.4f} | "
                f"set = 1 | heldout = {cfg.heldout_problem} | "
                f"solver = {cfg.solver} | surrogate = {base.effective_surrogate_label(cfg)} | "
                f"saea_steps = {cfg.saea_steps} | "
                f"eval_batch_size = {cfg.eval_batch_size} | workers = {workers} | "
                f"replay = {len(replay)} | reward_scheme = {cfg.reward_scheme} | "
                f"reward_norm = {int(cfg.reward_norm)}"
            )

            if cfg.cuda_cleanup_before_update:
                if cfg.debug:
                    log(f"[Epoch {epoch:04d}] nvidia-smi before cleanup/update | {base._nvidia_smi_snapshot()}")
                cleanup_stats = base._cleanup_cuda_cache(cfg.device, rounds=3, sleep_sec=1.0)
                if cfg.debug:
                    before = cleanup_stats["before"]
                    after = cleanup_stats["after"]
                    log(
                        f"[Epoch {epoch:04d}] cuda cleanup before update | "
                        f"free_mb={before['free_mb']:.2f}->{after['free_mb']:.2f} | "
                        f"alloc_mb={before['alloc_mb']:.2f}->{after['alloc_mb']:.2f} | "
                        f"reserved_mb={before['reserved_mb']:.2f}->{after['reserved_mb']:.2f}"
                    )
                    log(f"[Epoch {epoch:04d}] nvidia-smi after cleanup before update | {base._nvidia_smi_snapshot()}")
            pre_update_state_dict = base.clone_state_dict_cpu(agent)
            pre_update_cpu_mem = base._process_memory_stats_mb() if cfg.debug else None
            pre_update_cuda_mem = base._cuda_memory_stats_mb(cfg.device) if cfg.debug else None
            if cfg.debug and torch.cuda.is_available():
                try:
                    cuda_device = torch.device(str(cfg.device))
                    if cuda_device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(cuda_device)
                except Exception:
                    pass
            agent.train()
            target.eval()
            update_started = time.perf_counter()
            update_metrics = []
            if len(replay) >= TRAJECTORY_BATCH_SIZE:
                for _ in range(int(cfg.updates_per_epoch)):
                    update_metrics.append(update_from_trajectory_batch(agent, target, optimizer, replay, cfg))
            update_elapsed = time.perf_counter() - update_started
            post_update_cpu_mem = base._process_memory_stats_mb() if cfg.debug else None
            post_update_cuda_mem = base._cuda_memory_stats_mb(cfg.device) if cfg.debug else None

            if update_metrics:
                total_update_steps += len(update_metrics)
                averaged = {
                    key: float(np.mean([item[key] for item in update_metrics]))
                    for key in update_metrics[0]
                }
                log(
                    f"epoch {epoch} update done | replay={len(replay)} | updates={len(update_metrics)} | "
                    f"update_time_sec={update_elapsed:.3f} | td_loss={averaged['td_loss']:.6f} | "
                    f"grad_norm={averaged['grad_norm']:.6f} | q_mean={averaged['q_mean']:.4f} | "
                    f"target_mean={averaged['target_mean']:.4f} | "
                    f"td_error_mean={averaged['td_error_mean']:.4f} | "
                    f"trajectory_steps={averaged['trajectory_steps']:.2f}"
                )
                if cfg.debug:
                    log(
                        f"[Epoch {epoch:04d}] minibatch debug | "
                        f"updates={len(update_metrics)} | "
                        f"sample_cpu_total_sec={sum(item['sample_cpu_sec'] for item in update_metrics):.3f} | "
                        f"trajectory_batch_cpu_mb_mean={averaged['trajectory_batch_cpu_mb']:.2f} | "
                        f"state_input_mb_max={max(item['state_input_mb_max'] for item in update_metrics):.3f} | "
                        f"trajectory_steps_mean={averaged['trajectory_steps']:.2f} | "
                        f"candidate_rows_mean={averaged['candidate_rows_mean']:.2f} | "
                        f"candidate_rows_max={int(max(item['candidate_rows_max'] for item in update_metrics))}"
                    )
            else:
                log(f"epoch {epoch} update skipped | replay={len(replay)}")

            if total_update_steps - last_target_sync_update >= TARGET_SYNC_UPDATES:
                target.load_state_dict(agent.state_dict(), strict=True)
                last_target_sync_update = total_update_steps
                log(
                    f"epoch {epoch} target sync | sync_every_updates={TARGET_SYNC_UPDATES} | "
                    f"total_update_steps={total_update_steps} | replay={len(replay)}"
                )

            if cfg.debug and pre_update_cpu_mem is not None and post_update_cpu_mem is not None:
                log(
                    f"[Epoch {epoch:04d}] host_mem debug | "
                    f"rss_mb={post_update_cpu_mem['rss_mb']:.2f} | "
                    f"rss_delta_mb={post_update_cpu_mem['rss_mb'] - pre_update_cpu_mem['rss_mb']:.2f} | "
                    f"vms_mb={post_update_cpu_mem['vms_mb']:.2f} | "
                    f"uss_mb={post_update_cpu_mem['uss_mb']:.2f} | "
                    f"avail_mb={post_update_cpu_mem['avail_mb']:.2f}"
                )
            if cfg.debug and pre_update_cuda_mem is not None and post_update_cuda_mem is not None:
                log(
                    f"[Epoch {epoch:04d}] cuda_mem debug | "
                    f"alloc_mb={post_update_cuda_mem['alloc_mb']:.2f} | "
                    f"reserved_mb={post_update_cuda_mem['reserved_mb']:.2f} | "
                    f"peak_alloc_mb={post_update_cuda_mem['peak_alloc_mb']:.2f} | "
                    f"peak_reserved_mb={post_update_cuda_mem['peak_reserved_mb']:.2f} | "
                    f"alloc_delta_mb={post_update_cuda_mem['alloc_mb'] - pre_update_cuda_mem['alloc_mb']:.2f} | "
                    f"reserved_delta_mb={post_update_cuda_mem['reserved_mb'] - pre_update_cuda_mem['reserved_mb']:.2f} | "
                    f"free_mb={post_update_cuda_mem['free_mb']:.2f} | "
                    f"total_mb={post_update_cuda_mem['total_mb']:.2f}"
                )
                log(f"[Epoch {epoch:04d}] nvidia-smi after update | {base._nvidia_smi_snapshot()}")
            if cfg.cuda_cleanup_after_update:
                cleanup_stats = base._cleanup_cuda_cache(cfg.device, rounds=3, sleep_sec=1.0)
                if cfg.debug:
                    before = cleanup_stats["before"]
                    after = cleanup_stats["after"]
                    log(
                        f"[Epoch {epoch:04d}] cuda cleanup after update | "
                        f"free_mb={before['free_mb']:.2f}->{after['free_mb']:.2f} | "
                        f"alloc_mb={before['alloc_mb']:.2f}->{after['alloc_mb']:.2f} | "
                        f"reserved_mb={before['reserved_mb']:.2f}->{after['reserved_mb']:.2f}"
                    )
                    log(f"[Epoch {epoch:04d}] nvidia-smi after cleanup update | {base._nvidia_smi_snapshot()}")

            best_reward = base.save_training_checkpoint(
                agent,
                cfg,
                cfg.heldout_problem,
                epoch,
                mean_reward,
                best_reward,
                best_state_dict=pre_update_state_dict,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if ray_module is not None and ray_module.is_initialized():
            ray_module.shutdown()
        log_file.close()
    return agent


def main():
    args = base.parse_args(configure_parser=_configure_parser)
    args.agent_name = "boformer"
    args.reward_scheme = BOFORMER_REWARD_SCHEME
    args.reward_norm = False
    if not _has_cli_flag("--hidden_dim"):
        args.hidden_dim = 128
    if not _has_cli_flag("--gamma"):
        args.gamma = BOFORMER_GAMMA
    if not _has_cli_flag("--updates_per_epoch"):
        args.updates_per_epoch = BOFORMER_UPDATES_PER_EPOCH
    if not _has_cli_flag("--solver"):
        args.solver = "hybrid"
    train_boformer(args)


if __name__ == "__main__":
    main()
