from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from solver.solver import GPSurrogateProblem, _ModelListSurrogate
from solver.population_utils import sample_unique_non_parent, unique_non_parent_indices


@dataclass(frozen=True)
class NSGA3Result:
    x: np.ndarray
    y: np.ndarray


def _build_reference_directions(n_obj: int, pop_size: int, seed: int) -> np.ndarray:
    try:
        ref_dirs = get_reference_directions("energy", int(n_obj), int(pop_size), seed=int(seed))
        return np.asarray(ref_dirs, dtype=np.float64)
    except TypeError:
        ref_dirs = get_reference_directions("energy", int(n_obj), int(pop_size))
        return np.asarray(ref_dirs, dtype=np.float64)
    except Exception:
        pass

    n_partitions = 1
    while comb(int(n_partitions) + int(n_obj) - 1, int(n_obj) - 1) < int(pop_size):
        n_partitions += 1
    ref_dirs = get_reference_directions("das-dennis", int(n_obj), n_partitions=int(n_partitions))
    ref_dirs = np.asarray(ref_dirs, dtype=np.float64)
    if ref_dirs.shape[0] > int(pop_size):
        ref_dirs = ref_dirs[: int(pop_size)].copy()
    return ref_dirs


def _evaluate_surrogate(surrogate, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    if hasattr(surrogate, "predict_mean"):
        return np.asarray(surrogate.predict_mean(x_arr), dtype=np.float64)
    if hasattr(surrogate, "predict"):
        return np.asarray(surrogate.predict(x_arr), dtype=np.float64)
    if hasattr(surrogate, "evaluate"):
        val = surrogate.evaluate(x_arr)
        if isinstance(val, dict) and "F" in val:
            return np.asarray(val["F"], dtype=np.float64)
    raise TypeError("surrogate must implement predict_mean(x), predict(x), or evaluate(x)['F'].")


def _select_non_parent_offspring(
    *,
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    archive_x: np.ndarray,
    requested_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    requested_size = int(requested_size)
    candidate_x = np.asarray(candidate_x, dtype=np.float64)
    candidate_y = np.asarray(candidate_y, dtype=np.float64)
    selected_idx = unique_non_parent_indices(
        candidate_x,
        np.asarray(archive_x, dtype=np.float64),
        epsilon=1e-8,
        limit=requested_size,
    )
    if selected_idx.size == 0:
        return candidate_x[:0], candidate_y[:0]
    return candidate_x[selected_idx], candidate_y[selected_idx]


def _sample_random_non_parent(
    *,
    problem,
    archive_x: np.ndarray,
    existing_x: np.ndarray,
    n_samples: int,
    seed: int,
    max_attempt_factor: int = 100,
) -> np.ndarray:
    n_samples = int(n_samples)
    if n_samples <= 0:
        return np.empty((0, int(problem.n_var)), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    lower = np.asarray(problem.xl, dtype=np.float64)
    upper = np.asarray(problem.xu, dtype=np.float64)
    return sample_unique_non_parent(
        lower=lower,
        upper=upper,
        archive_x=archive_x,
        existing_x=existing_x,
        n_samples=n_samples,
        rng=rng,
        epsilon=1e-8,
        max_rounds=max(1, int(max_attempt_factor) // 5),
    ).astype(np.float64)


def run_surrogate_nsga3(
    problem,
    archive_x,
    pop_size,
    gps=None,
    surrogate=None,
    saea_steps=25,
    seed=0,
    n_gen=None,
    surrogate_nsga_steps=None,
):
    requested_pop_size = int(pop_size)
    internal_pop_size = max(requested_pop_size, int(4 * requested_pop_size))
    if surrogate_nsga_steps is not None:
        saea_steps = surrogate_nsga_steps
    if n_gen is not None:
        saea_steps = n_gen
    if surrogate is None:
        if gps is None:
            raise ValueError("run_surrogate_nsga3 requires either `surrogate` or `gps`.")
        surrogate = _ModelListSurrogate(gps)

    surrogate_problem = GPSurrogateProblem(
        surrogate=surrogate,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        xl=problem.xl,
        xu=problem.xu,
    )

    init_x = np.asarray(archive_x, dtype=np.float64)
    if init_x.shape[0] >= internal_pop_size:
        init_x = init_x[:internal_pop_size].copy()
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, init_x.shape[0], size=internal_pop_size)
        init_x = init_x[idx].copy()

    ref_dirs = _build_reference_directions(int(problem.n_obj), int(internal_pop_size), int(seed))
    algorithm = NSGA3(
        pop_size=int(internal_pop_size),
        ref_dirs=ref_dirs,
        sampling=init_x,
        eliminate_duplicates=True,
    )

    res = minimize(
        surrogate_problem,
        algorithm,
        termination=get_termination("n_gen", int(saea_steps)),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    candidate_x = np.asarray(res.X, dtype=np.float64)
    candidate_y = np.asarray(res.F, dtype=np.float64)
    if candidate_x.ndim == 1:
        candidate_x = candidate_x.reshape(1, -1)
    if candidate_y.ndim == 1:
        candidate_y = candidate_y.reshape(1, -1)
    selected_x, selected_y = _select_non_parent_offspring(
        candidate_x=candidate_x,
        candidate_y=candidate_y,
        archive_x=np.asarray(archive_x, dtype=np.float64),
        requested_size=requested_pop_size,
    )
    if int(selected_x.shape[0]) < requested_pop_size:
        n_missing = int(requested_pop_size) - int(selected_x.shape[0])
        random_x = _sample_random_non_parent(
            problem=problem,
            archive_x=np.asarray(archive_x, dtype=np.float64),
            existing_x=selected_x,
            n_samples=n_missing,
            seed=int(seed) + 104729,
        )
        random_y = _evaluate_surrogate(surrogate, random_x)
        selected_x = np.vstack([selected_x, random_x]).astype(np.float64)
        selected_y = np.vstack([selected_y, random_y]).astype(np.float64)
    return selected_x[:requested_pop_size], selected_y[:requested_pop_size]
