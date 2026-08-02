from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def latin_hypercube_sample(
    *,
    n_samples: int,
    dim: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    seed: int,
) -> np.ndarray:
    lower_arr = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper_arr = np.asarray(upper, dtype=np.float32).reshape(-1)
    if lower_arr.size == 1:
        lower_arr = np.repeat(lower_arr, int(dim))
    if upper_arr.size == 1:
        upper_arr = np.repeat(upper_arr, int(dim))

    sampler = qmc.LatinHypercube(d=int(dim), seed=int(seed))
    unit_samples = sampler.random(n=int(n_samples))
    return qmc.scale(unit_samples, lower_arr, upper_arr).astype(np.float32)


__all__ = ["latin_hypercube_sample"]
