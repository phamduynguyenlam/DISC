from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from solver.moead_ego_solver import run_surrogate_moead_ego
from solver.nsga3_solver import run_surrogate_nsga3
from surrogate.surrogate_model import estimate_uncertainty


@dataclass(frozen=True)
class HybridSolverResult:
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    source: np.ndarray


def _predict_mean(surrogate: Any, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_mean"):
        return np.asarray(surrogate.predict_mean(x_arr), dtype=np.float32)
    if hasattr(surrogate, "predict"):
        return np.asarray(surrogate.predict(x_arr), dtype=np.float32)
    raise TypeError("hybrid solver surrogate must implement predict_mean(x) or predict(x).")


def _predict_std(
    *,
    surrogate: Any,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(offspring_x, dtype=np.float32)
    y_arr = np.asarray(archive_y, dtype=np.float32)
    try:
        if hasattr(surrogate, "predict_std"):
            std = np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        elif hasattr(surrogate, "predict_mean_std"):
            _, std = surrogate.predict_mean_std(x_arr)
            std = np.asarray(std, dtype=np.float32)
        else:
            raise NotImplementedError
    except Exception:
        archive_pred = _predict_mean(surrogate, np.asarray(archive_x, dtype=np.float32))
        std = estimate_uncertainty(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=y_arr,
            archive_pred=archive_pred,
            offspring_x=x_arr,
        )

    if std.ndim == 1:
        std = std.reshape(-1, 1)
    if std.shape[1] == y_arr.shape[1]:
        return np.maximum(std, 1e-12).astype(np.float32)
    if std.shape[1] == 1:
        return np.repeat(std, y_arr.shape[1], axis=1).astype(np.float32)
    return np.zeros((int(x_arr.shape[0]), int(y_arr.shape[1])), dtype=np.float32)


def _unique_rows_keep_order(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    source: np.ndarray,
    *,
    max_rows: int,
    archive_x: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen: set[tuple[float, ...]] = set()
    keep: list[int] = []
    x_arr = np.asarray(x, dtype=np.float32)
    archive_arr = None if archive_x is None else np.asarray(archive_x, dtype=np.float32)
    for idx, row in enumerate(x_arr):
        key = tuple(np.round(row.astype(np.float64), decimals=8).tolist())
        if key in seen:
            continue
        if archive_arr is not None and archive_arr.size:
            if np.any(np.all(np.isclose(archive_arr, row[None, :], rtol=1e-8, atol=1e-8), axis=1)):
                continue
        seen.add(key)
        keep.append(int(idx))
        if len(keep) >= int(max_rows):
            break
    if len(keep) == 0:
        return x_arr[:0], mean[:0], std[:0], source[:0]
    idx_arr = np.asarray(keep, dtype=np.int64)
    return x_arr[idx_arr], mean[idx_arr], std[idx_arr], source[idx_arr]


def run_surrogate_hybrid(
    *,
    problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    surrogate: Any,
    pop_size: int = 80,
    saea_steps: int | None = None,
    seed: int = 0,
    nsga3_pop_size: int | None = None,
    moead_ego_pop_size: int | None = None,
    nsga3_saea_steps: int | None = None,
    moead_ego_saea_steps: int | None = None,
    surrogate_nsga_steps: int | None = None,
) -> HybridSolverResult:
    """Generate offspring by running NSGA-III and MOEA/D-EGO concurrently."""
    if surrogate_nsga_steps is not None:
        saea_steps = surrogate_nsga_steps
    nsga_steps = (
        int(nsga3_saea_steps)
        if nsga3_saea_steps is not None
        else (15 if saea_steps is None else int(saea_steps))
    )
    moead_steps = (
        int(moead_ego_saea_steps)
        if moead_ego_saea_steps is not None
        else (30 if saea_steps is None else int(saea_steps))
    )

    archive_x_arr = np.asarray(archive_x, dtype=np.float32)
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    total_size = max(1, int(pop_size))
    nsga_size = int(nsga3_pop_size) if nsga3_pop_size is not None else total_size // 2
    nsga_size = max(0, min(total_size, nsga_size))
    moead_size = int(moead_ego_pop_size) if moead_ego_pop_size is not None else total_size - nsga_size
    moead_size = max(0, min(total_size, moead_size))
    if nsga_size + moead_size <= 0:
        raise ValueError("hybrid solver requires at least one generated candidate.")

    def _run_nsga3() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if nsga_size <= 0:
            n_obj = int(archive_y_arr.shape[1])
            return (
                np.empty((0, int(archive_x_arr.shape[1])), dtype=np.float32),
                np.empty((0, n_obj), dtype=np.float32),
                np.empty((0, n_obj), dtype=np.float32),
                np.empty((0,), dtype=object),
            )
        x, mean = run_surrogate_nsga3(
            problem=problem,
            archive_x=archive_x_arr,
            pop_size=int(nsga_size),
            surrogate=surrogate,
            saea_steps=int(nsga_steps),
            seed=int(seed),
        )
        x_arr = np.asarray(x, dtype=np.float32)
        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = _predict_std(
            surrogate=surrogate,
            archive_x=archive_x_arr,
            archive_y=archive_y_arr,
            offspring_x=x_arr,
        )
        return x_arr, mean_arr, std_arr, np.full(int(x_arr.shape[0]), "nsga3", dtype=object)

    def _run_moead_ego() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if moead_size <= 0:
            n_obj = int(archive_y_arr.shape[1])
            return (
                np.empty((0, int(archive_x_arr.shape[1])), dtype=np.float32),
                np.empty((0, n_obj), dtype=np.float32),
                np.empty((0, n_obj), dtype=np.float32),
                np.empty((0,), dtype=object),
            )
        result = run_surrogate_moead_ego(
            problem=problem,
            archive_x=archive_x_arr,
            archive_y=archive_y_arr,
            surrogate=surrogate,
            pop_size=int(moead_size),
            saea_steps=int(moead_steps),
            seed=int(seed) + 7919,
        )
        x_arr = np.asarray(result.x, dtype=np.float32)
        return (
            x_arr,
            np.asarray(result.mean, dtype=np.float32),
            np.asarray(result.std, dtype=np.float32),
            np.full(int(x_arr.shape[0]), "moead_ego", dtype=object),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        nsga_future = executor.submit(_run_nsga3)
        moead_future = executor.submit(_run_moead_ego)
        nsga_x, nsga_mean, nsga_std, nsga_source = nsga_future.result()
        moead_x, moead_mean, moead_std, moead_source = moead_future.result()

    x = np.vstack([nsga_x, moead_x]).astype(np.float32)
    mean = np.vstack([nsga_mean, moead_mean]).astype(np.float32)
    std = np.vstack([nsga_std, moead_std]).astype(np.float32)
    source = np.concatenate([nsga_source, moead_source], axis=0)
    x, mean, std, source = _unique_rows_keep_order(
        x,
        mean,
        std,
        source,
        max_rows=total_size,
        archive_x=archive_x_arr,
    )
    return HybridSolverResult(x=x, mean=mean, std=std, source=source)


__all__ = [
    "HybridSolverResult",
    "run_surrogate_hybrid",
]
