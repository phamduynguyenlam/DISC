from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


DEFAULT_REWARD_LAMBDAS: dict[int, float] = {
    1: 5.0,
    2: 1.0,
}


def resolve_default_reward_lambda(reward_scheme: int) -> float:
    return float(DEFAULT_REWARD_LAMBDAS.get(int(reward_scheme), 5.0))


def pareto_front(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.reshape(0, 0).astype(np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values.shape}")

    keep = NonDominatedSorting().do(values, only_non_dominated_front=True)
    return values[np.asarray(keep, dtype=np.int64)]


def hypervolume(values: np.ndarray, ref_point: np.ndarray) -> float:
    front = pareto_front(values)
    if front.size == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref_point, dtype=np.float32))(front))


def _filter_candidates_on_front(
    selected_objectives: np.ndarray,
    combined_front: np.ndarray,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    selected_arr = np.asarray(selected_objectives, dtype=np.float32)
    if selected_arr.ndim == 1:
        selected_arr = selected_arr.reshape(1, -1)
    combined_arr = np.asarray(combined_front, dtype=np.float32)
    if combined_arr.size == 0:
        return selected_arr[:0]

    keep: list[np.ndarray] = []
    for candidate in selected_arr:
        matches = np.isclose(combined_arr, candidate[None, :], atol=float(atol), rtol=0.0)
        if bool(np.any(np.all(matches, axis=1))):
            keep.append(candidate)

    if len(keep) == 0:
        return np.empty((0, selected_arr.shape[1]), dtype=np.float32)
    return np.asarray(keep, dtype=np.float32)


def reward_scheme_1(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float = DEFAULT_REWARD_LAMBDAS[1],
) -> float:
    """Distance-to-front reward, scaled and offset; positive iff a new point stays on the updated Pareto front."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = pareto_front(np.vstack([previous_front, selected_objectives]))
    front_members = _filter_candidates_on_front(selected_objectives, combined_front)
    if front_members.shape[0] == 0:
        return -1.0

    if previous_front.size == 0:
        return float(max(1e-6, 1.0 + float(reward_lambda) * float(front_members.shape[0])))

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in front_members:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(max(1e-6, 1.0 + float(reward_lambda) * reward))


def reward_scheme_2(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    true_pareto_hv: float,
    hv_epsilon: float = 1e-8,
    reward_lambda: float = DEFAULT_REWARD_LAMBDAS[2],
) -> float:
    """Normalized HV improvement against the remaining gap to the true Pareto-front HV."""
    previous_front = np.asarray(previous_front, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = np.vstack([previous_front, selected_objectives])

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(combined_front, ref_point)
    if next_hv <= prev_hv:
        return 0.0
    remaining_gap = max(float(true_pareto_hv) - float(next_hv), float(hv_epsilon))
    return float(float(reward_lambda) * (float(next_hv) - float(prev_hv)) / remaining_gap)
