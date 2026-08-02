import os
import random
import sys
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import trainer as base_trainer
import torch
import torch.nn as nn
import numpy as np

from agents.db_saea import DBSAEAAgent
from solver.nsga3_solver import run_surrogate_nsga3
from problem.problem import get_reference_point, get_true_pareto_hv, make_problem
from lhs import latin_hypercube_sample as _scipy_latin_hypercube_sample
from reward import hypervolume, pareto_front, reward_scheme_2
from surrogate.surrogate_model import (
    estimate_uncertainty,
    fit_gp_surrogates,
)


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def resolve_default_reward_lambda(reward_scheme: int) -> float:
    scheme = int(reward_scheme)
    if scheme == 1:
        return 5.0
    if scheme == 2:
        return 1.0
    return 5.0
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@dataclass
class TrainConfig:
    seed: int = 0
    num_workers: int = 24
    episodes_per_worker: int = 1
    max_fe: int = 120
    init_size: int = 80
    eval_batch_size: int = 1
    batch_size: int = 64
    replay_size: int = 50000
    gamma: float = 0.99
    lr: float = 1e-4
    target_soft_tau: float = 1e-3
    train_iters: int = 50
    updates_per_epoch: int = 80
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_iters: int = 10
    hidden_dim: int = 16
    n_heads: int = 1
    ff_dim: int = 256
    dropout: float = 0.0
    logit_scale: float = 1.0
    surrogate_model: str = "gp"
    solver: str = "nsga3"
    saea_steps: int = 25
    saea_steps_user: bool = False
    offspring_size: int = 80
    reward_scheme: int = 1
    reward_lambda: float = 5.0
    reward_norm: bool = True
    policy_mode: str = "epsilon_greedy"
    heldout_problem: str = "ZDT1"
    weight_dir: str = "weight"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rollout_device: str = "cpu"
    surrogate_device: str = "cpu"
    agent_name: str = "db_saea"
    prediction_history_steps: int = 40
    nsga_af: str = "mean"
    beta: float = 1.0
    hybrid_nsga3_steps: int | None = None
    hybrid_moead_ego_steps: int | None = None
    cuda_cleanup_before_update: bool = False
    cuda_cleanup_after_update: bool = False


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def extend(self, transitions):
        self.buffer.extend(transitions)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


def epsilon_by_iter(it, cfg):
    frac = min(1.0, it / cfg.epsilon_decay_iters)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def clone_state_dict_cpu(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def soft_update_target_network(target_model, online_model, tau: float) -> None:
    tau_value = float(tau)
    one_minus_tau = 1.0 - tau_value
    with torch.no_grad():
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.mul_(one_minus_tau).add_(online_param.data, alpha=tau_value)
        for target_buffer, online_buffer in zip(target_model.buffers(), online_model.buffers()):
            target_buffer.copy_(online_buffer)


def resolve_agent_cls(agent_name):
    name = str(agent_name).strip().lower()
    if name == "db_saea":
        return DBSAEAAgent
    raise ValueError(f"Unsupported agent_name for db_saea_trainer: {agent_name}")


def select_action_from_output(out):
    if "action" in out:
        return int(out["action"].reshape(-1)[0].item())
    ranking = out["ranking"]
    return int(ranking[0, 0].item())


def parse_args():
    return base_trainer.parse_args()


TRAIN_PROBLEM_POOL = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]


def latin_hypercube_sample(n_samples, dim, lower, upper, seed):
    return _scipy_latin_hypercube_sample(
        n_samples=int(n_samples), dim=int(dim), lower=lower, upper=upper, seed=int(seed)
    )


def build_named_surrogate_from_cfg(cfg_dict, archive_x, archive_y, surrogate_name, existing_surrogate=None):
    del existing_surrogate
    surrogate_name = str(surrogate_name).lower()

    if surrogate_name == "gp":
        return fit_gp_surrogates(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=np.asarray(archive_y, dtype=np.float32),
            seed=int(cfg_dict.get("seed", 0)),
            nu=float(cfg_dict.get("gp_nu", 5.0)),
        )

    raise ValueError(f"Unsupported surrogate_model: {surrogate_name}")


def build_surrogate_from_cfg(cfg_dict, archive_x, archive_y, existing_surrogate=None):
    surrogate_name = str(cfg_dict.get("surrogate_model", "gp")).lower()
    return build_named_surrogate_from_cfg(
        cfg_dict,
        archive_x,
        archive_y,
        surrogate_name,
        existing_surrogate=existing_surrogate,
    )


def surrogate_or_models_for_nsga2(surrogate):
    models = getattr(surrogate, "models", None)
    if isinstance(models, list) and len(models) > 0:
        return None, models
    return surrogate, None


def load_true_pareto_front(
    problem_name: str,
    dim: int,
    n_obj: int,
    n_points: int = 400,
) -> np.ndarray | None:
    try:
        problem = make_problem(problem_name, dim=int(dim))
        if hasattr(problem, "pareto_front"):
            pareto = problem.pareto_front(n_pareto_points=int(n_points))
            if pareto is not None:
                pareto = np.asarray(pareto, dtype=np.float32)
                if pareto.ndim == 2 and pareto.shape[1] >= int(n_obj):
                    return pareto[:, : int(n_obj)]
    except Exception:
        pass

    try:
        from pymoo.problems import get_problem
    except Exception:
        return None

    key = str(problem_name).lower()
    try:
        pymoo_problem = get_problem(key, n_var=int(dim), n_obj=int(n_obj))
    except TypeError:
        try:
            pymoo_problem = get_problem(key, n_var=int(dim))
        except Exception:
            return None
    except Exception:
        return None

    try:
        pareto = pymoo_problem.pareto_front(n_pareto_points=int(n_points))
    except TypeError:
        try:
            pareto = pymoo_problem.pareto_front()
        except Exception:
            return None
    except Exception:
        return None

    if pareto is None:
        return None
    pareto = np.asarray(pareto, dtype=np.float32)
    if pareto.ndim != 2 or pareto.shape[1] < int(n_obj):
        return None
    return pareto[:, : int(n_obj)]


def compute_true_pareto_hv(problem_name: str, dim: int, ref_point: np.ndarray, n_obj: int) -> float:
    try:
        cached_true_hv = get_true_pareto_hv(str(problem_name), dim=int(dim), n_obj=int(n_obj))
    except Exception:
        cached_true_hv = None
    if cached_true_hv is not None:
        return float(cached_true_hv)

    true_pareto = load_true_pareto_front(problem_name, int(dim), int(n_obj))
    if true_pareto is None:
        raise RuntimeError(
            f"Could not load true Pareto front or cached true Pareto HV: "
            f"problem={problem_name}, dim={dim}, n_obj={n_obj}."
        )
    return float(hypervolume(true_pareto, np.asarray(ref_point, dtype=np.float32)))


def make_nsga2_problem_adapter(problem, n_obj):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def predict_surrogate_mean(surrogate, x):
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(surrogate, x):
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_std"):
        try:
            return np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        except NotImplementedError:
            pass
    return np.zeros((int(x_arr.shape[0]), 1), dtype=np.float32)


def build_offspring_sigma(archive_x, archive_y, offspring_x, surrogate):
    archive_y = np.asarray(archive_y, dtype=np.float32)
    sigma = predict_surrogate_std(surrogate, offspring_x)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)
    if sigma.shape[1] == archive_y.shape[1]:
        return sigma.astype(np.float32)

    archive_pred = predict_surrogate_mean(surrogate, archive_x)
    local_sigma = estimate_uncertainty(
        archive_x=np.asarray(archive_x, dtype=np.float32),
        archive_y=archive_y,
        archive_pred=archive_pred,
        offspring_x=np.asarray(offspring_x, dtype=np.float32),
    )
    if local_sigma.ndim == 1:
        local_sigma = local_sigma.reshape(-1, 1)
    if local_sigma.shape[1] != archive_y.shape[1]:
        local_sigma = np.repeat(local_sigma.mean(axis=1, keepdims=True), archive_y.shape[1], axis=1)
    return local_sigma.astype(np.float32)


def _generate_single_offspring_pool(cfg_dict, archive_x, archive_y, nsga2_problem, seed, surrogate_name):
    surrogate = build_named_surrogate_from_cfg(
        cfg_dict,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate_name=surrogate_name,
    )
    nsga2_surrogate, nsga2_models = surrogate_or_models_for_nsga2(surrogate)
    solver_name = str(cfg_dict.get("solver", "nsga3")).lower()
    if solver_name != "nsga3":
        raise ValueError(f"Unsupported trainer solver: {solver_name}")
    offspring_x, offspring_y = run_surrogate_nsga3(
        gps=nsga2_models,
        surrogate=nsga2_surrogate,
        problem=nsga2_problem,
        archive_x=archive_x,
        pop_size=int(cfg_dict["offspring_size"]),
        saea_steps=int(cfg_dict["saea_steps"]),
        seed=int(seed),
    )
    offspring_x = np.asarray(offspring_x, dtype=np.float32)
    offspring_y = np.asarray(offspring_y, dtype=np.float32)
    offspring_sigma = build_offspring_sigma(
        archive_x=archive_x,
        archive_y=archive_y,
        offspring_x=offspring_x,
        surrogate=surrogate,
    )
    return surrogate, offspring_x, offspring_y, offspring_sigma


def generate_offspring_pool(cfg_dict, archive_x, archive_y, nsga2_problem, seed):
    return _generate_single_offspring_pool(
        cfg_dict,
        archive_x=archive_x,
        archive_y=archive_y,
        nsga2_problem=nsga2_problem,
        seed=seed,
        surrogate_name=str(cfg_dict.get("surrogate_model", "gp")).lower(),
    )


def pad_stack_rows(arrays, pad_value=0.0):
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty.")

    max_ndim = max(arr.ndim for arr in arrays)
    normalized = []
    for arr in arrays:
        if arr.ndim < max_ndim:
            new_shape = arr.shape + (1,) * (max_ndim - arr.ndim)
            arr = arr.reshape(new_shape)
        normalized.append(arr)

    target_shape = tuple(
        max(int(arr.shape[axis]) for arr in normalized)
        for axis in range(max_ndim)
    )

    padded = []
    for arr in normalized:
        if arr.shape == target_shape:
            padded.append(arr)
            continue

        out = np.full(target_shape, pad_value, dtype=np.float32)
        slices = tuple(slice(0, int(size)) for size in arr.shape)
        out[slices] = arr
        padded.append(out)

    return np.stack(padded, axis=0)


def build_row_mask(arrays):
    arrays = [np.asarray(arr) for arr in arrays]
    max_rows = max(int(arr.shape[0]) for arr in arrays)
    mask = np.zeros((len(arrays), max_rows), dtype=bool)
    for i, arr in enumerate(arrays):
        mask[i, : int(arr.shape[0])] = True
    return mask


def _compute_ddqn_loss_same_objectives(agent, target_agent, batch, cfg):
    (
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
    ) = batch

    device = cfg.device
    batch_to_device_started_at = time.perf_counter()
    archive_mask = torch.as_tensor(build_row_mask(x_true), dtype=torch.bool, device=device)
    candidate_mask = torch.as_tensor(build_row_mask(x_sur), dtype=torch.bool, device=device)
    next_archive_mask = torch.as_tensor(build_row_mask(next_x_true), dtype=torch.bool, device=device)
    next_candidate_mask = torch.as_tensor(build_row_mask(next_x_sur), dtype=torch.bool, device=device)

    x_true = to_tensor(pad_stack_rows(x_true), device)
    y_true = to_tensor(pad_stack_rows(y_true), device)
    x_sur = to_tensor(pad_stack_rows(x_sur), device)
    y_sur = to_tensor(pad_stack_rows(y_sur), device)
    sigma_sur = to_tensor(pad_stack_rows(sigma_sur), device)
    progress = to_tensor(np.asarray(progress).reshape(-1, 1), device)

    lower_bound = to_tensor(pad_stack_rows(lower_bound), device)
    upper_bound = to_tensor(pad_stack_rows(upper_bound), device)

    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    next_x_true = to_tensor(pad_stack_rows(next_x_true), device)
    next_y_true = to_tensor(pad_stack_rows(next_y_true), device)
    next_x_sur = to_tensor(pad_stack_rows(next_x_sur), device)
    next_y_sur = to_tensor(pad_stack_rows(next_y_sur), device)
    next_sigma_sur = to_tensor(pad_stack_rows(next_sigma_sur), device)
    next_progress = to_tensor(np.asarray(next_progress).reshape(-1, 1), device)
    batch_to_device_sec = time.perf_counter() - batch_to_device_started_at

    out = agent(
        x_true=x_true,
        y_true=y_true,
        x_sur=x_sur,
        y_sur=y_sur,
        sigma_sur=sigma_sur,
        progress=progress,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        archive_mask=archive_mask,
        candidate_mask=candidate_mask,
        decode_type="q_greedy",
    )

    q_values = out["q_values"]
    q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        next_target = target_agent(
            x_true=next_x_true,
            y_true=next_y_true,
            x_sur=next_x_sur,
            y_sur=next_y_sur,
            sigma_sur=next_sigma_sur,
            progress=next_progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=next_archive_mask,
            candidate_mask=next_candidate_mask,
            decode_type="q_greedy",
        )
        next_q = next_target["q_values"].max(dim=1).values
        target = rewards + cfg.gamma * next_q * (1.0 - dones)

    td_error = q_sa - target
    loss = nn.SmoothL1Loss()(q_sa, target)
    metrics = {
        "q_mean": q_sa.detach().mean().item(),
        "q_std": q_sa.detach().std(unbiased=False).item() if q_sa.numel() > 1 else 0.0,
        "target_mean": target.detach().mean().item(),
        "td_error_mean": td_error.detach().mean().item(),
        "td_error_std": td_error.detach().std(unbiased=False).item() if td_error.numel() > 1 else 0.0,
        "reward_mean": rewards.mean().item(),
        "batch_to_device_sec": float(batch_to_device_sec),
    }
    return loss, metrics, len(x_true)


def compute_ddqn_loss(agent, target_agent, batch, cfg):
    (
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
    ) = batch

    objective_counts = [int(np.asarray(arr).shape[1]) for arr in y_true]
    groups = {}
    for idx, n_obj in enumerate(objective_counts):
        groups.setdefault(n_obj, []).append(idx)

    total_count = 0
    total_q_mean = 0.0
    total_q_std = 0.0
    total_target_mean = 0.0
    total_td_error_mean = 0.0
    total_td_error_std = 0.0
    total_r_mean = 0.0
    total_batch_to_device_sec = 0.0
    weighted_loss = None
    group_sizes = []
    group_objectives = []

    batch_items = [
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
    ]

    for n_obj, indices in groups.items():
        subbatch = []
        for item in batch_items:
            subbatch.append([item[i] for i in indices])

        group_loss, group_metrics, group_count = _compute_ddqn_loss_same_objectives(
            agent=agent,
            target_agent=target_agent,
            batch=tuple(subbatch),
            cfg=cfg,
        )

        total_count += int(group_count)
        group_sizes.append(int(group_count))
        group_objectives.append(int(n_obj))
        total_q_mean += float(group_metrics["q_mean"]) * float(group_count)
        total_q_std += float(group_metrics["q_std"]) * float(group_count)
        total_target_mean += float(group_metrics["target_mean"]) * float(group_count)
        total_td_error_mean += float(group_metrics["td_error_mean"]) * float(group_count)
        total_td_error_std += float(group_metrics["td_error_std"]) * float(group_count)
        total_r_mean += float(group_metrics["reward_mean"]) * float(group_count)
        total_batch_to_device_sec += float(group_metrics["batch_to_device_sec"])

        scaled_loss = group_loss * (float(group_count) / float(len(objective_counts)))
        weighted_loss = scaled_loss if weighted_loss is None else (weighted_loss + scaled_loss)

    if weighted_loss is None or total_count <= 0:
        raise ValueError("Failed to build objective-shape groups for DDQN loss.")

    metrics = {
        "q_mean": total_q_mean / total_count,
        "q_std": total_q_std / total_count,
        "target_mean": total_target_mean / total_count,
        "td_error_mean": total_td_error_mean / total_count,
        "td_error_std": total_td_error_std / total_count,
        "reward_mean": total_r_mean / total_count,
        "batch_to_device_sec": total_batch_to_device_sec,
        "shape_group": len(group_sizes),
        "group_sizes": group_sizes,
        "group_objectives": group_objectives,
        "shape_group_detail": {
            int(n_obj): int(sz) for n_obj, sz in zip(group_objectives, group_sizes)
        },
    }
    return weighted_loss, metrics


def compute_env_reward(
    previous_archive_y,
    selected_y,
    ref_point,
    reward_scheme_id,
    reward_lambda=5.0,
    true_pareto_hv=None,
):
    previous_front = pareto_front(np.asarray(previous_archive_y, dtype=np.float32))
    selected_y = np.asarray(selected_y, dtype=np.float32)

    if int(reward_scheme_id) == 1:
        if previous_front.size == 0:
            return float(max(1e-6, 1.0 + float(reward_lambda) * float(selected_y.shape[0])))
        improved_mask = np.asarray(
            [
                not any(
                    np.all(prev <= candidate) and np.any(prev < candidate)
                    for prev in previous_front
                )
                for candidate in selected_y
            ],
            dtype=bool,
        )
        improved_candidates = np.asarray(selected_y[improved_mask], dtype=np.float32)
        if improved_candidates.shape[0] <= 0:
            return -1.0
        reward = 1.0
        origin = np.zeros(previous_front.shape[1], dtype=np.float32)
        for candidate in improved_candidates:
            distances = np.abs(previous_front - candidate).sum(axis=1)
            nearest_idx = int(np.argmin(distances))
            d_i = float(distances[nearest_idx])
            d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
            reward += float(reward_lambda) * d_i / max(d_ref_i, 1e-12)
        return float(max(1e-6, reward))
    if int(reward_scheme_id) == 2:
        if true_pareto_hv is None:
            raise ValueError("reward_scheme_2 requires true_pareto_hv.")
        return float(
            reward_scheme_2(
                previous_front=previous_front,
                selected_objectives=selected_y,
                ref_point=ref_point,
                true_pareto_hv=float(true_pareto_hv),
                reward_lambda=float(reward_lambda),
            )
        )
    raise ValueError(f"Unsupported reward_scheme: {reward_scheme_id}")


def env_key(problem_name, dim):
    return f"{str(problem_name).upper()}-{int(dim)}D"


class DiscSAEAEnv:
    def __init__(self, problem_name, dim, seed, cfg_dict):
        self.problem_name = str(problem_name)
        self.dim = int(dim)
        self.seed = int(seed)
        self.cfg = dict(cfg_dict)
        self.problem = make_problem(self.problem_name, dim=self.dim)
        self.lower_bound = np.full(self.dim, float(self.problem.lower), dtype=np.float32)
        self.upper_bound = np.full(self.dim, float(self.problem.upper), dtype=np.float32)
        self.max_steps = max(1, int(self.cfg["max_fe"]) - int(self.cfg["init_size"]))
        self.t = 0
        self.archive_x = None
        self.archive_y = None
        self.offspring_x = None
        self.offspring_y = None
        self.offspring_sigma = None
        self.ref_point = None
        self.nsga2_problem = None
        self.init_hv = None
        self.true_pareto_hv = None
        self.surrogate = None
        self._surrogate_dirty = False
        self._regenerate_counter = 0

    def _progress(self):
        max_fe = int(self.cfg["max_fe"])
        current_fe = min(int(self.cfg["init_size"]) + int(self.t), max_fe - 1)
        return float(current_fe) / float(max_fe)

    def _surrogate_cfg(self):
        cfg_local = dict(self.cfg)
        cfg_local["seed"] = int(self.seed) + int(self.t)
        return cfg_local

    def _fit_surrogate(self):
        self.surrogate = build_surrogate_from_cfg(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
        )
        self._surrogate_dirty = False
        return self.surrogate

    def _ensure_surrogate_ready(self):
        if self.surrogate is None or bool(self._surrogate_dirty):
            self._fit_surrogate()
        return self.surrogate

    def _refresh_offspring(self):
        surrogate, offspring_x, offspring_y, offspring_sigma = generate_offspring_pool(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
            nsga2_problem=self.nsga2_problem,
            seed=int(self.seed) + int(self.t) + 97 * int(self._regenerate_counter),
        )
        self.surrogate = surrogate
        self.offspring_x = np.asarray(offspring_x, dtype=np.float32)
        self.offspring_y = np.asarray(offspring_y, dtype=np.float32)
        self.offspring_sigma = np.asarray(offspring_sigma, dtype=np.float32)

    def _build_state(self):
        return {
            "x_true": np.asarray(self.archive_x, dtype=np.float32).copy(),
            "y_true": np.asarray(self.archive_y, dtype=np.float32).copy(),
            "x_sur": np.asarray(self.offspring_x, dtype=np.float32).copy(),
            "y_sur": np.asarray(self.offspring_y, dtype=np.float32).copy(),
            "sigma_sur": np.asarray(self.offspring_sigma, dtype=np.float32).copy(),
            "progress": np.array([self._progress()], dtype=np.float32),
            "lower_bound": self.lower_bound.copy(),
            "upper_bound": self.upper_bound.copy(),
        }

    def reset(self):
        self.t = 0
        self._regenerate_counter = 0
        self.archive_x = latin_hypercube_sample(
            n_samples=int(self.cfg["init_size"]),
            dim=self.dim,
            lower=self.problem.lower,
            upper=self.problem.upper,
            seed=self.seed,
        )
        self.archive_y = np.asarray(self.problem.evaluate(self.archive_x), dtype=np.float32)
        self.ref_point = np.asarray(
            get_reference_point(self.problem_name, n_obj=int(self.archive_y.shape[1])),
            dtype=np.float32,
        )
        self.init_hv = float(hypervolume(self.archive_y, self.ref_point))
        self.true_pareto_hv = compute_true_pareto_hv(
            self.problem_name,
            self.dim,
            self.ref_point,
            int(self.archive_y.shape[1]),
        )
        self.nsga2_problem = make_nsga2_problem_adapter(self.problem, int(self.archive_y.shape[1]))
        # Pretrain surrogate on the initial archive (default: 80 points), then generate first offspring pool.
        self._fit_surrogate()
        self._refresh_offspring()
        return self._build_state()

    def step(self, action_idx, state):
        del state
        chosen_idx = int(np.clip(int(action_idx), 0, int(self.offspring_x.shape[0]) - 1))
        previous_archive_y = np.asarray(self.archive_y, dtype=np.float32).copy()
        chosen_x = self.offspring_x[chosen_idx : chosen_idx + 1]
        chosen_y = np.asarray(self.problem.evaluate(chosen_x), dtype=np.float32)

        self.archive_x = np.vstack([self.archive_x, chosen_x]).astype(np.float32)
        self.archive_y = np.vstack([self.archive_y, chosen_y]).astype(np.float32)

        reward = compute_env_reward(
            previous_archive_y=previous_archive_y,
            selected_y=chosen_y,
            ref_point=self.ref_point,
            reward_scheme_id=int(self.cfg["reward_scheme"]),
            reward_lambda=float(self.cfg.get("reward_lambda", 5.0)),
            true_pareto_hv=self.true_pareto_hv,
        )

        self.t += 1
        self._regenerate_counter = 0
        done = self.t >= self.max_steps
        self._surrogate_dirty = bool(not done)
        if not done:
            self._refresh_offspring()
        return self._build_state(), float(reward), bool(done)

    def regenerate_offspring(self):
        self._regenerate_counter += 1
        self._refresh_offspring()
        return self._build_state(), 0.0, False

    def evaluate_selected_index(self, selected_idx: int):
        chosen_idx = int(np.clip(int(selected_idx), 0, int(self.offspring_x.shape[0]) - 1))
        previous_archive_y = np.asarray(self.archive_y, dtype=np.float32).copy()
        chosen_x = self.offspring_x[chosen_idx : chosen_idx + 1]
        chosen_y = np.asarray(self.problem.evaluate(chosen_x), dtype=np.float32)

        self.archive_x = np.vstack([self.archive_x, chosen_x]).astype(np.float32)
        self.archive_y = np.vstack([self.archive_y, chosen_y]).astype(np.float32)

        reward = compute_env_reward(
            previous_archive_y=previous_archive_y,
            selected_y=chosen_y,
            ref_point=self.ref_point,
            reward_scheme_id=int(self.cfg["reward_scheme"]),
            reward_lambda=float(self.cfg.get("reward_lambda", 5.0)),
            true_pareto_hv=self.true_pareto_hv,
        )

        self.t += 1
        self._regenerate_counter = 0
        done = self.t >= self.max_steps
        self._surrogate_dirty = bool(not done)
        if not done:
            self._refresh_offspring()
        return self._build_state(), float(reward), bool(done)

    def current_hv(self):
        return float(hypervolume(np.asarray(self.archive_y, dtype=np.float32), self.ref_point))


def _rollout_episode_impl(state_dict_cpu, cfg_dict, problem_name, dim, seed, epsilon):
    task_started_at = time.perf_counter()
    worker_pid = os.getpid()
    device = str(cfg_dict.get("rollout_device", "cpu"))
    agent_cls = resolve_agent_cls(cfg_dict.get("agent_name", "db_saea"))
    agent = agent_cls(
        hidden_dim=cfg_dict["hidden_dim"],
        n_heads=cfg_dict["n_heads"],
        ff_dim=cfg_dict["ff_dim"],
        dropout=cfg_dict["dropout"],
        logit_scale=cfg_dict["logit_scale"],
        epsilon=epsilon,
    ).to(device)

    agent.load_state_dict(state_dict_cpu)
    agent.eval()

    env = DiscSAEAEnv(
        problem_name=problem_name,
        dim=int(dim),
        seed=int(seed),
        cfg_dict=cfg_dict,
    )
    state = env.reset()
    total_reward = 0.0
    init_hv = float(env.init_hv)
    transitions = []
    true_eval_reward = 0.0
    true_eval_count = 0

    done = False
    while not done:
        with torch.no_grad():
            out = agent(
                x_true=to_tensor(state["x_true"][None, ...], device),
                y_true=to_tensor(state["y_true"][None, ...], device),
                x_sur=to_tensor(state["x_sur"][None, ...], device),
                y_sur=to_tensor(state["y_sur"][None, ...], device),
                sigma_sur=to_tensor(state["sigma_sur"][None, ...], device),
                progress=to_tensor(state["progress"].reshape(1, 1), device),
                lower_bound=to_tensor(state["lower_bound"][None, ...], device),
                upper_bound=to_tensor(state["upper_bound"][None, ...], device),
                decode_type=str(cfg_dict.get("policy_mode", "epsilon_greedy")),
                epsilon=epsilon,
            )

        action = select_action_from_output(out)
        if int(action) == 0:
            next_state, reward, done = env.regenerate_offspring()
        else:
            selected_idx, _ = agent.select_candidate_from_action(
                action_idx=int(action),
                archive_y=state["y_true"],
                candidate_mean=state["y_sur"],
                candidate_std=state["sigma_sur"],
                seed=int(seed) + len(transitions),
            )
            if selected_idx is None:
                raise RuntimeError(f"DB-SAEA action {action} did not produce a candidate index.")
            next_state, reward, done = env.evaluate_selected_index(int(selected_idx))
            true_eval_reward += float(reward)
            true_eval_count += 1

        transitions.append((
            state["x_true"],
            state["y_true"],
            state["x_sur"],
            state["y_sur"],
            state["sigma_sur"],
            float(state["progress"][0]),
            state["lower_bound"],
            state["upper_bound"],
            action,
            reward,
            next_state["x_true"],
            next_state["y_true"],
            next_state["x_sur"],
            next_state["y_sur"],
            next_state["sigma_sur"],
            float(next_state["progress"][0]),
            float(done),
        ))

        total_reward += reward
        state = next_state

    if bool(cfg_dict.get("reward_norm", False)):
        transitions = _normalize_episode_transition_rewards(transitions)

    task_finished_at = time.perf_counter()
    return {
        "transitions": transitions,
        "episode_reward": float(total_reward),
        "episode_steps": int(env.t),
        "logging_reward": float(true_eval_reward),
        "logging_steps": int(true_eval_count),
        "env_key": env_key(problem_name, dim),
        "init_hv": init_hv,
        "final_hv": float(env.current_hv()),
        "worker_pid": int(worker_pid),
        "task_started_at": float(task_started_at),
        "task_finished_at": float(task_finished_at),
        "task_duration_sec": float(task_finished_at - task_started_at),
    }


def rollout_episode_task_local(state_dict_cpu, cfg_dict, problem_name, dim, seed, epsilon):
    return _rollout_episode_impl(
        state_dict_cpu=state_dict_cpu,
        cfg_dict=cfg_dict,
        problem_name=problem_name,
        dim=dim,
        seed=seed,
        epsilon=epsilon,
    )


def _normalize_episode_transition_rewards(transitions):
    if len(transitions) == 0:
        return transitions

    rewards = np.asarray([float(transition[9]) for transition in transitions], dtype=np.float32)
    reward_mean = float(np.mean(rewards))
    reward_std = float(np.std(rewards))
    denom = reward_std if reward_std > 1e-8 else 1.0
    normalized_rewards = (rewards - reward_mean) / denom

    normalized_transitions = []
    for idx, transition in enumerate(transitions):
        transition_list = list(transition)
        transition_list[9] = float(normalized_rewards[idx])
        normalized_transitions.append(tuple(transition_list))
    return normalized_transitions


def summarize_parallel_rollout(results):
    if not results:
        return None
    started = np.asarray([float(r["task_started_at"]) for r in results], dtype=np.float64)
    finished = np.asarray([float(r["task_finished_at"]) for r in results], dtype=np.float64)
    durations = np.asarray([float(r["task_duration_sec"]) for r in results], dtype=np.float64)
    pids = [int(r["worker_pid"]) for r in results]
    overlap_wall_sec = float(max(finished.max() - started.min(), 0.0))
    sum_task_sec = float(durations.sum())
    parallelism = float(sum_task_sec / max(overlap_wall_sec, 1e-12))
    return {
        "tasks": int(len(results)),
        "unique_pids": int(len(set(pids))),
        "pid_list": sorted(set(pids)),
        "overlap_wall_sec": overlap_wall_sec,
        "sum_task_sec": sum_task_sec,
        "mean_task_sec": float(durations.mean()),
        "min_task_sec": float(durations.min()),
        "max_task_sec": float(durations.max()),
        "parallelism_est": parallelism,
    }


def _prepare_training_run_layout(log_dir: str, log_filename: str) -> tuple[str, str]:
    os.makedirs(log_dir, exist_ok=True)

    run_stem = os.path.splitext(log_filename)[0]
    run_dir = os.path.join(log_dir, run_stem)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, os.path.join(run_dir, log_filename)


def save_training_checkpoint(
    agent,
    cfg,
    problem_name,
    epoch,
    mean_reward,
    best_reward,
    best_state_dict=None,
):
    os.makedirs(cfg.weight_dir, exist_ok=True)
    rs_tag = f"rs{int(cfg.reward_scheme)}"
    hidden_tag = f"h{int(cfg.hidden_dim)}"
    problem_tag = str(problem_name).lower()
    agent_tag = str(getattr(cfg, "agent_name", "db_saea")).lower()
    file_prefix = f"{agent_tag}_problem_{problem_tag}_{rs_tag}_{hidden_tag}"

    if mean_reward > best_reward:
        best_path = os.path.join(cfg.weight_dir, f"{file_prefix}_best_reward.pth")
        state_dict_to_save = best_state_dict if best_state_dict is not None else agent.state_dict()
        torch.save(
            {
                "epoch": int(epoch),
                "problem_name": str(problem_name),
                "reward_scheme": int(cfg.reward_scheme),
                "mean_reward": float(mean_reward),
                "state_dict": state_dict_to_save,
            },
            best_path,
        )
        best_reward = float(mean_reward)

    epoch_path = os.path.join(cfg.weight_dir, f"{file_prefix}_epoch_{int(epoch)}.pth")
    torch.save(
        {
            "epoch": int(epoch),
            "problem_name": str(problem_name),
            "reward_scheme": int(cfg.reward_scheme),
            "mean_reward": float(mean_reward),
            "state_dict": agent.state_dict(),
        },
        epoch_path,
    )

    return best_reward


def train_db_saea_ddqn_ray(
    problem_name="ZDT1",
    dim=30,
    seed=0,
    epoch=None,
    gamma=0.99,
    reward_scheme=1,
    reward_norm=True,
    surrogate_model="gp",
    solver="hybrid",
    hybrid_nsga3_steps=None,
    hybrid_moead_ego_steps=None,
    num_workers=None,
    eval_batch_size=1,
    updates_per_epoch=None,
    hidden_dim=None,
    device=None,
    rollout_device="cpu",
    surrogate_device="cpu",
    cuda_cleanup_before_update=True,
    cuda_cleanup_after_update=True,
    disable_dual_control=False,
    use_ray=False,
):
    return base_trainer.train_disc_ddqn_ray(
        problem_name=problem_name,
        dim=dim,
        seed=seed,
        epoch=epoch,
        gamma=gamma,
        reward_scheme=reward_scheme,
        reward_norm=reward_norm,
        surrogate_model=surrogate_model,
        solver=solver,
        hybrid_nsga3_steps=hybrid_nsga3_steps,
        hybrid_moead_ego_steps=hybrid_moead_ego_steps,
        num_workers=num_workers,
        eval_batch_size=eval_batch_size,
        updates_per_epoch=updates_per_epoch,
        hidden_dim=hidden_dim,
        device=device,
        rollout_device=rollout_device,
        surrogate_device=surrogate_device,
        disable_dual_control=disable_dual_control,
        use_ray=use_ray,
        agent_name="db_saea",
        cuda_cleanup_before_update=cuda_cleanup_before_update,
        cuda_cleanup_after_update=cuda_cleanup_after_update,
    )


if __name__ == "__main__":
    args = base_trainer.parse_args()
    train_db_saea_ddqn_ray(
        problem_name=args.problem,
        dim=int(args.dim),
        seed=int(args.seed),
        epoch=args.epoch,
        gamma=args.gamma,
        reward_scheme=int(args.reward_scheme),
        reward_norm=bool(args.reward_norm),
        surrogate_model=str(args.surrogate_model),
        solver=str(args.solver),
        hybrid_nsga3_steps=args.hybrid_nsga3_steps,
        hybrid_moead_ego_steps=args.hybrid_moead_ego_steps,
        num_workers=args.num_workers,
        eval_batch_size=int(args.batch),
        updates_per_epoch=args.updates_per_epoch,
        hidden_dim=args.hidden_dim,
        device=args.device,
        rollout_device=str(args.rollout_device),
        surrogate_device=str(args.surrogate_device),
        cuda_cleanup_before_update=(
            True if args.cuda_cleanup_before_update is None else bool(args.cuda_cleanup_before_update)
        ),
        cuda_cleanup_after_update=(
            True if args.cuda_cleanup_after_update is None else bool(args.cuda_cleanup_after_update)
        ),
        disable_dual_control=bool(args.disable_dual_control),
        use_ray=bool(args.ray),
    )
