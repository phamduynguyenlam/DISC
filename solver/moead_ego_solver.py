from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from pymoo.operators.mutation.pm import mut_pm
from pymoo.util.ref_dirs import get_reference_directions

from lhs import latin_hypercube_sample
from solver.population_utils import sample_unique_non_parent, unique_non_parent_indices


@dataclass(frozen=True)
class MOEADEGOSolverResult:
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    eti: np.ndarray
    weight_vectors: np.ndarray


def _problem_bounds(problem, archive_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(archive_x, dtype=np.float32)
    dim = int(x_arr.shape[1])
    lower = np.asarray(getattr(problem, "xl", np.zeros(dim)), dtype=np.float32).reshape(-1)
    upper = np.asarray(getattr(problem, "xu", np.ones(dim)), dtype=np.float32).reshape(-1)
    if lower.size == 1:
        lower = np.repeat(lower, dim)
    if upper.size == 1:
        upper = np.repeat(upper, dim)
    return lower.astype(np.float32), upper.astype(np.float32)


def _generate_weights(n_obj: int, n_weights: int, seed: int) -> np.ndarray:
    if int(n_obj) <= 1:
        return np.ones((int(n_weights), 1), dtype=np.float32)
    try:
        weights = get_reference_directions(
            "energy", int(n_obj), int(n_weights), seed=int(seed)
        )
    except TypeError:
        weights = get_reference_directions("energy", int(n_obj), int(n_weights))
    return np.maximum(weights, 1e-6).astype(np.float32)


def _tchebycheff(y: np.ndarray, weights: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.max(weights[:, None, :] * (y[None, :, :] - z[None, None, :]), axis=2)


def _polynomial_mutation(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    eta: float = 20.0,
    prob: float | None = None,
) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64).reshape(1, -1)
    dim = int(y.shape[1])
    mut_prob = 1.0 / float(max(dim, 1)) if prob is None else float(prob)
    mutated = mut_pm(
        y,
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        eta=np.full(1, float(eta), dtype=np.float64),
        prob=np.full(1, mut_prob, dtype=np.float64),
        at_least_once=False,
        random_state=rng,
    )
    return np.asarray(mutated[0], dtype=np.float32)


def _predict_mean_std(surrogate: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_mean"):
        mean = surrogate.predict_mean(x_arr)
    elif hasattr(surrogate, "predict"):
        mean = surrogate.predict(x_arr)
    else:
        raise TypeError("surrogate must implement predict_mean(x) or predict(x).")
    if hasattr(surrogate, "predict_std"):
        std = surrogate.predict_std(x_arr)
    elif hasattr(surrogate, "predict_mean_std"):
        _, std = surrogate.predict_mean_std(x_arr)
    else:
        raise TypeError("MOEA/D-EGO solver requires surrogate uncertainty via predict_std(x).")
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.maximum(np.asarray(std, dtype=np.float32), 1e-12)
    if mean_arr.ndim != 2 or std_arr.ndim != 2:
        raise ValueError(f"surrogate mean/std must be 2D, got {mean_arr.shape} and {std_arr.shape}.")
    return mean_arr, std_arr


def _select_unique_non_parent(
    *,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eti: np.ndarray,
    weights: np.ndarray,
    archive_x: np.ndarray,
    requested_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = unique_non_parent_indices(
        np.asarray(x, dtype=np.float64),
        np.asarray(archive_x, dtype=np.float64),
        epsilon=1e-8,
        limit=int(requested_size),
    )
    if idx.size == 0:
        n_obj = int(np.asarray(mean).shape[1])
        return (
            np.empty((0, int(np.asarray(x).shape[1])), dtype=np.float32),
            np.empty((0, n_obj), dtype=np.float32),
            np.empty((0, n_obj), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, n_obj), dtype=np.float32),
        )
    return (
        np.asarray(x, dtype=np.float32)[idx],
        np.asarray(mean, dtype=np.float32)[idx],
        np.asarray(std, dtype=np.float32)[idx],
        np.asarray(eti, dtype=np.float32)[idx],
        np.asarray(weights, dtype=np.float32)[idx],
    )


def _sample_unique_non_parent(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    archive_x: np.ndarray,
    existing_x: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return sample_unique_non_parent(
        lower=lower,
        upper=upper,
        archive_x=archive_x,
        existing_x=existing_x,
        n_samples=int(n_samples),
        rng=rng,
        epsilon=1e-8,
    ).astype(np.float32)


def _max_two_gaussians(
    *,
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu1_arr = np.asarray(mu1, dtype=np.float64)
    mu2_arr = np.asarray(mu2, dtype=np.float64)
    sigma1_arr = np.maximum(np.asarray(sigma1, dtype=np.float64), 1e-12)
    sigma2_arr = np.maximum(np.asarray(sigma2, dtype=np.float64), 1e-12)
    tau = np.sqrt(np.maximum(sigma1_arr ** 2 + sigma2_arr ** 2, 1e-24))
    alpha = (mu1_arr - mu2_arr) / tau
    phi = norm.pdf(alpha)
    Phi = norm.cdf(alpha)
    mean = mu1_arr * Phi + mu2_arr * (1.0 - Phi) + tau * phi
    second = (
        (mu1_arr ** 2 + sigma1_arr ** 2) * Phi
        + (mu2_arr ** 2 + sigma2_arr ** 2) * (1.0 - Phi)
        + (mu1_arr + mu2_arr) * tau * phi
    )
    var = np.maximum(second - mean ** 2, 1e-12)
    return np.asarray(mean, dtype=np.float32), np.asarray(np.sqrt(var), dtype=np.float32)


def _tchebycheff_gaussian_stats(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean, dtype=np.float64)
    std_arr = np.maximum(np.asarray(std, dtype=np.float64), 1e-12)
    weights_arr = np.asarray(weights, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64).reshape(1, -1)
    weighted_mean = weights_arr * (mean_arr - z_arr)
    weighted_std = weights_arr * std_arr
    if weighted_mean.shape[1] == 1:
        return (
            np.asarray(weighted_mean[:, 0], dtype=np.float32),
            np.asarray(np.maximum(weighted_std[:, 0], 1e-12), dtype=np.float32),
        )

    agg_mean = np.asarray(weighted_mean[:, 0], dtype=np.float32)
    agg_std = np.asarray(np.maximum(weighted_std[:, 0], 1e-12), dtype=np.float32)
    for obj_idx in range(1, weighted_mean.shape[1]):
        agg_mean, agg_std = _max_two_gaussians(
            mu1=agg_mean,
            sigma1=agg_std,
            mu2=weighted_mean[:, obj_idx],
            sigma2=weighted_std[:, obj_idx],
        )
    return agg_mean, np.maximum(agg_std, 1e-12).astype(np.float32)


def tchebycheff_gaussian_stats(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate the Gaussian moments of candidate-wise Tchebycheff values."""
    return _tchebycheff_gaussian_stats(
        mean=mean,
        std=std,
        weights=weights,
        z=z,
    )


def _expected_tchebycheff_improvement(
    *,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    gmin: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    g_mean, g_std = _tchebycheff_gaussian_stats(
        mean=mean,
        std=std,
        weights=weights,
        z=z,
    )
    gmin_arr = np.asarray(gmin, dtype=np.float64).reshape(-1)
    g_mean_arr = np.asarray(g_mean, dtype=np.float64).reshape(-1)
    g_std_arr = np.maximum(np.asarray(g_std, dtype=np.float64).reshape(-1), 1e-12)
    u = (gmin_arr - g_mean_arr) / g_std_arr
    ei = (gmin_arr - g_mean_arr) * norm.cdf(u) + g_std_arr * norm.pdf(u)
    return np.asarray(np.maximum(ei, 0.0), dtype=np.float32)


def run_surrogate_moead_ego(
    *,
    problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    surrogate: Any,
    pop_size: int | None = None,
    saea_steps: int = 30,
    seed: int = 0,
    neighbor_frac: float = 0.1,
    delta: float = 0.9,
    nr: int = 2,
    de_f: float = 0.5,
    de_cr: float = 1.0,
    duplicate_tol: float = 1e-6,
    surrogate_nsga_steps: int | None = None,
) -> MOEADEGOSolverResult:
    if surrogate_nsga_steps is not None:
        saea_steps = surrogate_nsga_steps
    archive_x_arr = np.asarray(archive_x, dtype=np.float32)
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    if archive_x_arr.ndim != 2 or archive_y_arr.ndim != 2:
        raise ValueError("archive_x and archive_y must be 2D arrays.")
    if archive_x_arr.shape[0] != archive_y_arr.shape[0]:
        raise ValueError("archive_x and archive_y row counts must match.")

    rng = np.random.default_rng(int(seed))
    lower, upper = _problem_bounds(problem, archive_x_arr)
    effective_pop_size = int(archive_x_arr.shape[0]) if pop_size is None else int(pop_size)
    if effective_pop_size < 2:
        raise ValueError(f"MOEA/D-EGO pop_size must be at least 2, got {effective_pop_size}.")

    weights = _generate_weights(int(archive_y_arr.shape[1]), effective_pop_size, int(seed))
    z = np.min(archive_y_arr, axis=0) - 1e-6 * np.maximum(np.std(archive_y_arr, axis=0), 1.0)
    gmin = np.min(_tchebycheff(archive_y_arr, weights, z), axis=1).astype(np.float32)

    n_neighbors = max(2, int(np.ceil(float(neighbor_frac) * effective_pop_size)))
    n_neighbors = min(n_neighbors, max(2, effective_pop_size))
    neighbors = np.argsort(cdist(weights, weights), axis=1)[:, :n_neighbors]

    pop_x = latin_hypercube_sample(
        n_samples=effective_pop_size,
        dim=int(archive_x_arr.shape[1]),
        lower=lower,
        upper=upper,
        seed=int(rng.integers(1_000_000_000)),
    ).astype(np.float32)
    pop_mean, pop_std = _predict_mean_std(surrogate, pop_x)
    pop_eti = _expected_tchebycheff_improvement(
        mean=pop_mean,
        std=pop_std,
        weights=weights,
        gmin=gmin,
        z=z,
    )

    for _ in range(max(1, int(saea_steps))):
        for i in range(effective_pop_size):
            pool = neighbors[i] if float(rng.random()) < float(delta) else np.arange(effective_pop_size)
            if len(pool) < 2:
                pool = np.arange(effective_pop_size)
            p = rng.choice(pool, size=2, replace=False)

            trial = pop_x[i].copy()
            mask = rng.random(int(archive_x_arr.shape[1])) < float(de_cr)
            trial[mask] = pop_x[i, mask] + float(de_f) * (pop_x[p[0], mask] - pop_x[p[1], mask])
            trial = np.clip(trial, lower, upper).astype(np.float32)
            trial = _polynomial_mutation(trial, lower, upper, rng)

            trial_mean, trial_std = _predict_mean_std(surrogate, trial[None, :])
            cand_weights = weights[pool]
            cand_gmin = gmin[pool]
            trial_mean_rep = np.repeat(trial_mean, len(pool), axis=0)
            trial_std_rep = np.repeat(trial_std, len(pool), axis=0)
            trial_eti = _expected_tchebycheff_improvement(
                mean=trial_mean_rep,
                std=trial_std_rep,
                weights=cand_weights,
                gmin=cand_gmin,
                z=z,
            )
            improve_local = np.where(trial_eti > pop_eti[pool])[0]
            if len(improve_local) <= 0:
                continue
            chosen = improve_local[: int(nr)]
            idx = pool[chosen]
            pop_x[idx] = trial
            pop_mean[idx] = trial_mean[0]
            pop_std[idx] = trial_std[0]
            pop_eti[idx] = trial_eti[chosen]

    pop_x, pop_mean, pop_std, pop_eti, pop_weights = _select_unique_non_parent(
        x=pop_x,
        mean=pop_mean,
        std=pop_std,
        eti=pop_eti,
        weights=weights,
        archive_x=archive_x_arr,
        requested_size=effective_pop_size,
    )
    if int(pop_x.shape[0]) < int(effective_pop_size):
        n_missing = int(effective_pop_size) - int(pop_x.shape[0])
        fill_x = _sample_unique_non_parent(
            lower=lower,
            upper=upper,
            archive_x=archive_x_arr,
            existing_x=pop_x,
            n_samples=n_missing,
            rng=rng,
        )
        fill_mean, fill_std = _predict_mean_std(surrogate, fill_x)
        fill_weights = np.asarray(weights[:n_missing], dtype=np.float32)
        fill_gmin = np.min(
            _tchebycheff(archive_y_arr, fill_weights, z),
            axis=1,
        ).astype(np.float32)
        fill_eti = _expected_tchebycheff_improvement(
            mean=fill_mean,
            std=fill_std,
            weights=fill_weights,
            gmin=fill_gmin,
            z=z,
        )
        pop_x = np.vstack([pop_x, fill_x]).astype(np.float32)
        pop_mean = np.vstack([pop_mean, fill_mean]).astype(np.float32)
        pop_std = np.vstack([pop_std, fill_std]).astype(np.float32)
        pop_eti = np.concatenate([pop_eti, fill_eti]).astype(np.float32)
        pop_weights = np.vstack([pop_weights, fill_weights]).astype(np.float32)

    return MOEADEGOSolverResult(
        x=np.asarray(pop_x, dtype=np.float32),
        mean=np.asarray(pop_mean, dtype=np.float32),
        std=np.asarray(pop_std, dtype=np.float32),
        eti=np.asarray(pop_eti, dtype=np.float32),
        weight_vectors=np.asarray(pop_weights, dtype=np.float32),
    )


__all__ = [
    "MOEADEGOSolverResult",
    "run_surrogate_moead_ego",
    "tchebycheff_gaussian_stats",
]
