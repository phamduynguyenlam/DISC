from __future__ import annotations

import numpy as np
from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.population import Population


def unique_non_parent_indices(
    candidate_x: np.ndarray,
    archive_x: np.ndarray,
    *,
    epsilon: float = 1e-8,
    limit: int | None = None,
) -> np.ndarray:
    candidates = np.asarray(candidate_x, dtype=np.float64)
    archive = np.asarray(archive_x, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError(f"candidate_x must be 2D, got shape={candidates.shape}.")
    if candidates.shape[0] == 0:
        return np.empty(0, dtype=np.int64)

    population = Population.new("X", candidates)
    references = Population.new("X", archive) if archive.size else Population.empty()
    _, keep, _ = DefaultDuplicateElimination(epsilon=float(epsilon)).do(
        population,
        references,
        return_indices=True,
    )
    indices = np.asarray(keep, dtype=np.int64)
    if limit is not None:
        indices = indices[: max(0, int(limit))]
    return indices


def sample_unique_non_parent(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    archive_x: np.ndarray,
    existing_x: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    epsilon: float = 1e-8,
    max_rounds: int = 20,
) -> np.ndarray:
    lower_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
    archive = np.asarray(archive_x, dtype=np.float64).reshape(-1, lower_arr.size)
    existing = np.asarray(existing_x, dtype=np.float64).reshape(-1, lower_arr.size)
    requested = max(0, int(n_samples))
    selected = np.empty((0, lower_arr.size), dtype=np.float64)

    for _ in range(max(1, int(max_rounds))):
        missing = requested - int(selected.shape[0])
        if missing <= 0:
            break
        proposal = rng.uniform(lower_arr, upper_arr, size=(max(32, 4 * missing), lower_arr.size))
        references = np.vstack([archive, existing, selected])
        keep = unique_non_parent_indices(
            proposal,
            references,
            epsilon=float(epsilon),
            limit=missing,
        )
        if keep.size:
            selected = np.vstack([selected, proposal[keep]])

    if int(selected.shape[0]) < requested:
        missing = requested - int(selected.shape[0])
        selected = np.vstack(
            [selected, rng.uniform(lower_arr, upper_arr, size=(missing, lower_arr.size))]
        )
    return selected[:requested]


__all__ = ["sample_unique_non_parent", "unique_non_parent_indices"]
