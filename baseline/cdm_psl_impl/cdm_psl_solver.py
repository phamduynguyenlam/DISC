from __future__ import annotations

import gc
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


def _install_pymoo_factory_shim() -> None:
    if importlib.util.find_spec("pymoo.factory") is not None:
        return

    module = types.ModuleType("pymoo.factory")

    def get_performance_indicator(name: str, **kwargs):
        if name != "hv":
            raise NotImplementedError(
                f"Only hv is supported by the compatibility shim, got {name!r}."
            )

        from pymoo.indicators.hv import HV

        hv = HV(ref_point=np.asarray(kwargs["ref_point"], dtype=float))

        class _HVWrapper:
            def __init__(self, indicator):
                self._indicator = indicator

            def calc(self, values):
                return self._indicator(np.asarray(values, dtype=float))

        return _HVWrapper(hv)

    module.get_performance_indicator = get_performance_indicator
    sys.modules["pymoo.factory"] = module


_ROOT_DIR = Path(__file__).resolve().parents[1]
_CDM_PSL_DIR = _ROOT_DIR / "cdm_psl"
_CDM_PSL_PATH = str(_CDM_PSL_DIR)

_cdm_psl_path_inserted = False
if _CDM_PSL_PATH not in sys.path:
    sys.path.insert(0, _CDM_PSL_PATH)
    _cdm_psl_path_inserted = True

_install_pymoo_factory_shim()

from baseline.cdm_psl_impl.diffusion import gen_offspring
from baseline.cdm_psl_impl.evolution.dom import pareto_dominance
from baseline.cdm_psl_impl.evolution.utils import init_dom_rel_map
from baseline.cdm_psl_impl.learning.model_init import init_dom_nn_classifier
from baseline.cdm_psl_impl.learning.prediction import nn_predict_dom_intra
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from surrogate.surrogate_model import fit_gp_surrogates
from baseline.cdm_psl_impl.utils import environment_selection, pm_mutation, sbx, sort_population

if _cdm_psl_path_inserted:
    sys.path.remove(_CDM_PSL_PATH)


@dataclass(frozen=True)
class _ArchiveProblem:
    n_var: int
    n_obj: int


def _cleanup_cuda_cache() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


class _RepoSurrogateAdapter:
    def __init__(self, surrogate, lower: np.ndarray, upper: np.ndarray, eps: float = 1e-4):
        self.surrogate = surrogate
        self.lower = np.asarray(lower, dtype=np.float32).reshape(1, -1)
        self.upper = np.asarray(upper, dtype=np.float32).reshape(1, -1)
        self.span = np.maximum(self.upper - self.lower, 1e-12)
        self.eps = float(eps)

    def _denormalize(self, x_norm: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_norm, dtype=np.float32)
        return (self.lower + x_arr * self.span).astype(np.float32)

    def _predict_mean(self, x_norm: np.ndarray) -> np.ndarray:
        x_raw = self._denormalize(x_norm)
        if hasattr(self.surrogate, "predict_mean"):
            return np.asarray(self.surrogate.predict_mean(x_raw), dtype=np.float32)
        if hasattr(self.surrogate, "predict"):
            return np.asarray(self.surrogate.predict(x_raw), dtype=np.float32)
        raise TypeError("CDM-PSL surrogate must implement predict_mean(x) or predict(x).")

    def _predict_std(self, x_norm: np.ndarray) -> np.ndarray:
        x_raw = self._denormalize(x_norm)
        if hasattr(self.surrogate, "predict_std"):
            return np.asarray(self.surrogate.predict_std(x_raw), dtype=np.float32)
        if hasattr(self.surrogate, "predict_mean_std"):
            _, std = self.surrogate.predict_mean_std(x_raw)
            return np.asarray(std, dtype=np.float32)
        raise TypeError("CDM-PSL surrogate requires uncertainty via predict_std(x).")

    def _finite_diff_gradient(self, x_norm: np.ndarray, predict_fn) -> np.ndarray:
        x_arr = np.asarray(x_norm, dtype=np.float32)
        base_shape = predict_fn(x_arr[:1]).shape
        n_obj = int(base_shape[1])
        grad = np.zeros((int(x_arr.shape[0]), n_obj, int(x_arr.shape[1])), dtype=np.float32)
        for dim_idx in range(int(x_arr.shape[1])):
            plus = x_arr.copy()
            minus = x_arr.copy()
            plus[:, dim_idx] = np.clip(plus[:, dim_idx] + self.eps, 0.0, 1.0)
            minus[:, dim_idx] = np.clip(minus[:, dim_idx] - self.eps, 0.0, 1.0)
            denom = np.maximum((plus[:, dim_idx] - minus[:, dim_idx]).reshape(-1, 1), 1e-12)
            grad[:, :, dim_idx] = (predict_fn(plus) - predict_fn(minus)) / denom
        return grad

    def evaluate(
        self,
        x,
        std: bool = False,
        calc_gradient: bool = False,
        calc_hessian: bool = False,
    ) -> dict:
        del calc_hessian
        x_norm = np.asarray(x, dtype=np.float32)
        mean = self._predict_mean(x_norm)
        out = {"F": mean}
        if std:
            out["S"] = np.maximum(self._predict_std(x_norm), 1e-12)
        if calc_gradient:
            out["dF"] = self._finite_diff_gradient(x_norm, self._predict_mean)
            if std:
                out["dS"] = self._finite_diff_gradient(x_norm, self._predict_std)
        return out


class CDMPSLQuerySolver:


    def __init__(
        self,
        n_select: int = 1,
        coef_lcb: float = 0.1,
        use_diffusion: bool = True,
        dominance_pool_ratio: float = 1.0 / 3.0,
        relation_map_size: int = 300,
        sbx_rounds: int = 1000,
        diffusion_augmentation_factor: int = 10,
        diffusion_num_steps: int = 25,
        diffusion_batch_size: int = 1024,
        diffusion_num_epoch: int = 4000,
        diffusion_guided_samples: int = 10,
        diffusion_random_samples: int = 100,
        seed: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_select = int(n_select)
        self.coef_lcb = float(coef_lcb)
        self.use_diffusion = bool(use_diffusion)
        self.dominance_pool_ratio = float(dominance_pool_ratio)
        self.relation_map_size = int(relation_map_size)
        self.sbx_rounds = int(sbx_rounds)
        self.diffusion_augmentation_factor = int(diffusion_augmentation_factor)
        self.diffusion_num_steps = int(diffusion_num_steps)
        self.diffusion_batch_size = int(diffusion_batch_size)
        self.diffusion_num_epoch = int(diffusion_num_epoch)
        self.diffusion_guided_samples = int(diffusion_guided_samples)
        self.diffusion_random_samples = int(diffusion_random_samples)
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def query(
        self,
        archive_x: Sequence[Sequence[float]] | np.ndarray,
        archive_y: Sequence[Sequence[float]] | np.ndarray,
        xl: Optional[Sequence[float] | np.ndarray] = None,
        xu: Optional[Sequence[float] | np.ndarray] = None,
        surrogate=None,
    ) -> dict:
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        X_raw, Y = self._validate_archive(archive_x, archive_y)
        X_norm, lower, upper = self._normalize_archive(X_raw, xl, xu)

        n_archive, n_var = X_norm.shape
        n_obj = Y.shape[1]
        problem = _ArchiveProblem(n_var=n_var, n_obj=n_obj)

        if surrogate is None:
            surrogate = fit_gp_surrogates(
                archive_x=X_raw.astype(np.float32),
                archive_y=Y.astype(np.float32),
                seed=0 if self.seed is None else int(self.seed),
            )
        surrogate_model = _RepoSurrogateAdapter(surrogate, lower=lower, upper=upper)

        p_rel_map, _ = init_dom_rel_map(max(self.relation_map_size, n_archive))
        p_model = init_dom_nn_classifier(
            X_norm, Y, p_rel_map, pareto_dominance, problem
        )

        elite_size = self._get_elite_size(n_archive)
        _, elite_idx = environment_selection(Y, elite_size)
        parent_pool = X_norm[elite_idx, :]
        if len(parent_pool) < 2:
            parent_pool = X_norm.copy()

        domination_model_trained = p_model is not None
        if domination_model_trained:
            try:
                dom_labels, dom_conf = nn_predict_dom_intra(
                    parent_pool, p_model, self.device
                )
                sorted_pop = sort_population(parent_pool, dom_labels, dom_conf)
            except Exception:
                domination_model_trained = False
                sorted_pop = parent_pool
            finally:
                if p_model is not None:
                    try:
                        p_model.to("cpu")
                    except Exception:
                        pass
                    del p_model
                    _cleanup_cuda_cache()
        else:
            sorted_pop = parent_pool

        sorted_pop = self._ensure_min_parent_pool(np.asarray(sorted_pop, dtype=float))
        lbound = torch.zeros(n_var, dtype=torch.float32)
        ubound = torch.ones(n_var, dtype=torch.float32)

        if self.use_diffusion:
            offspring_norm = gen_offspring(
                sorted_pop,
                n_var,
                surrogate_model=surrogate_model,
                boundary=[lbound, ubound],
                augmentation_factor=self.diffusion_augmentation_factor,
                num_steps=self.diffusion_num_steps,
                batch_size=self.diffusion_batch_size,
                num_epoch=self.diffusion_num_epoch,
                n_guided_samples=self.diffusion_guided_samples,
                n_random_samples=self.diffusion_random_samples,
                coef_lcb=self.coef_lcb,
            )
        else:
            offspring_norm = self._generate_sbx_offspring(sorted_pop)

        offspring_norm = pm_mutation(
            np.asarray(offspring_norm, dtype=float), [lbound, ubound]
        )
        offspring_norm = np.clip(offspring_norm, 0.0, 1.0)

        pred = surrogate_model.evaluate(offspring_norm, std=True)
        pred_mean_raw = np.asarray(pred["F"], dtype=float)
        pred_std_raw = np.asarray(pred["S"], dtype=float)

        valid_mask = ~np.any(np.isnan(pred_mean_raw), axis=1)
        valid_mask &= ~np.any(np.isnan(pred_std_raw), axis=1)
        offspring_norm = offspring_norm[valid_mask]
        pred_mean_raw = pred_mean_raw[valid_mask]
        pred_std_raw = pred_std_raw[valid_mask]

        if len(offspring_norm) == 0:
            raise RuntimeError("CDM-PSL did not produce any valid offspring.")

        candidate_lcb = pred_mean_raw - self.coef_lcb * pred_std_raw
        nd_idx = NonDominatedSorting().do(Y)
        archive_nd = Y[nd_idx[0]]
        selected_idx = self._greedy_hv_select(
            archive_nd=archive_nd,
            candidate_y=candidate_lcb,
            n_select=min(self.n_select, len(offspring_norm)),
        )

        offspring_raw = self._denormalize(offspring_norm, lower, upper)
        selected_norm = offspring_norm[selected_idx]
        selected_raw = offspring_raw[selected_idx]
        result = {
            "archive_x_normalized": X_norm,
            "archive_y": Y,
            "offspring_x_normalized": offspring_norm,
            "offspring_x": offspring_raw,
            "offspring_pred_mean_normalized": pred_mean_raw,
            "offspring_pred_std_normalized": pred_std_raw,
            "offspring_pred_mean": pred_mean_raw,
            "offspring_pred_std": pred_std_raw,
            "offspring_pred_lcb_normalized": candidate_lcb,
            "selected_indices": selected_idx,
            "selected_x_normalized": selected_norm,
            "selected_x": selected_raw,
            "selected_pred_mean_normalized": pred_mean_raw[selected_idx],
            "selected_pred_std_normalized": pred_std_raw[selected_idx],
            "selected_pred_mean": pred_mean_raw[selected_idx],
            "selected_pred_std": pred_std_raw[selected_idx],
            "selected_pred_lcb_normalized": candidate_lcb[selected_idx],
            "domination_model_trained": domination_model_trained,
            "used_diffusion": self.use_diffusion,
        }
        _cleanup_cuda_cache()
        return result

    def _generate_sbx_offspring(self, sorted_pop: np.ndarray) -> np.ndarray:
        rows_to_take = max(2, int(self.dominance_pool_ratio * sorted_pop.shape[0]))
        offspring_a = sorted_pop[:rows_to_take, :]

        if len(offspring_a) % 2 == 1:
            offspring_a = offspring_a[:-1]
        if len(offspring_a) < 2:
            offspring_a = self._ensure_min_parent_pool(sorted_pop)[:2]

        new_pop = np.empty((0, sorted_pop.shape[1]), dtype=float)
        for _ in range(self.sbx_rounds):
            new_pop = np.vstack((new_pop, sbx(offspring_a, eta=15)))
        return new_pop

    def _greedy_hv_select(
        self, archive_nd: np.ndarray, candidate_y: np.ndarray, n_select: int
    ) -> np.ndarray:
        chosen = []
        remaining = list(range(len(candidate_y)))
        current_front = np.asarray(archive_nd, dtype=float)

        for _ in range(n_select):
            if not remaining:
                break

            ref_point = np.max(np.vstack([current_front, candidate_y[remaining]]), axis=0)
            hv = HV(ref_point=ref_point)
            best_idx = None
            best_hv_value = -np.inf

            for idx in remaining:
                hv_value = hv(np.vstack([current_front, candidate_y[idx]]))
                if hv_value > best_hv_value:
                    best_hv_value = hv_value
                    best_idx = idx

            if best_idx is None:
                break

            chosen.append(best_idx)
            current_front = np.vstack([current_front, candidate_y[best_idx]])
            remaining.remove(best_idx)

        return np.asarray(chosen, dtype=int)

    def _get_elite_size(self, n_archive: int) -> int:
        elite_size = int(self.dominance_pool_ratio * n_archive)
        elite_size = max(2, elite_size)
        elite_size = min(n_archive, elite_size)
        return elite_size

    def _ensure_min_parent_pool(
        self, parent_pool: np.ndarray, minimum_size: int = 6
    ) -> np.ndarray:
        if len(parent_pool) >= minimum_size:
            return parent_pool

        repeats = int(np.ceil(minimum_size / max(len(parent_pool), 1)))
        expanded = np.tile(parent_pool, (repeats, 1))
        return expanded[:minimum_size]

    def _validate_archive(
        self,
        archive_x: Sequence[Sequence[float]] | np.ndarray,
        archive_y: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(archive_x, dtype=float)
        Y = np.asarray(archive_y, dtype=float)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("archive_x and archive_y must both be 2D arrays.")
        if len(X) != len(Y):
            raise ValueError("archive_x and archive_y must have the same length.")
        if len(X) < 2:
            raise ValueError("At least 2 archive points are required.")

        return X, Y

    def _normalize_archive(
        self,
        X: np.ndarray,
        xl: Optional[Sequence[float] | np.ndarray],
        xu: Optional[Sequence[float] | np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if xl is None and xu is None:
            if np.min(X) < -1e-8 or np.max(X) > 1.0 + 1e-8:
                raise ValueError(
                    "archive_x must already be normalized to [0, 1] when xl/xu are not provided."
                )
            return X.copy(), np.zeros(X.shape[1]), np.ones(X.shape[1])
        if xl is None or xu is None:
            raise ValueError("xl and xu must either both be provided or both be None.")

        lower = np.asarray(xl, dtype=float)
        upper = np.asarray(xu, dtype=float)

        if lower.ndim == 0:
            lower = np.full(X.shape[1], float(lower))
        if upper.ndim == 0:
            upper = np.full(X.shape[1], float(upper))
        if lower.shape != (X.shape[1],) or upper.shape != (X.shape[1],):
            raise ValueError("xl and xu must have shape (n_var,).")
        if np.any(upper <= lower):
            raise ValueError("Each element of xu must be strictly larger than xl.")

        X_norm = (X - lower) / (upper - lower)
        X_norm = np.clip(X_norm, 0.0, 1.0)
        return X_norm, lower, upper

    def _denormalize(
        self, X_norm: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        if np.allclose(lower, 0.0) and np.allclose(upper, 1.0):
            return X_norm.copy()
        return X_norm * (upper - lower) + lower


def query_cdm_psl(
    archive_x: Sequence[Sequence[float]] | np.ndarray,
    archive_y: Sequence[Sequence[float]] | np.ndarray,
    **solver_kwargs,
) -> dict:
    """
    Convenience wrapper around `CDMPSLQuerySolver(...).query(...)`.

    Example:
        result = query_cdm_psl(archive_x, archive_y, n_select=5)
        offspring = result["offspring_x"]
        next_batch = result["selected_x"]
    """

    xl = solver_kwargs.pop("xl", None)
    xu = solver_kwargs.pop("xu", None)
    surrogate = solver_kwargs.pop("surrogate", None)
    solver = CDMPSLQuerySolver(**solver_kwargs)
    return solver.query(archive_x=archive_x, archive_y=archive_y, xl=xl, xu=xu, surrogate=surrogate)


def run_surrogate_cdm_psl(
    *,
    problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    surrogate=None,
    pop_size: int | None = None,
    seed: int = 0,
    coef_lcb: float = 0.1,
    sbx_rounds: int = 100,
    n_select: int = 1,
    diffusion_augmentation_factor: int = 10,
    diffusion_num_steps: int = 25,
    diffusion_batch_size: int = 1024,
    diffusion_num_epoch: int = 4000,
    diffusion_guided_samples: int = 10,
    diffusion_random_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate CDM-PSL solver offspring with diffusion guidance.

    The SBX/PM branch remains available through CDMPSLQuerySolver/query_cdm_psl
    for future baselines, but the solver interface is intentionally diffusion-only.
    """

    archive_x_arr = np.asarray(archive_x, dtype=np.float32)
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    lower = np.asarray(getattr(problem, "xl", np.zeros(archive_x_arr.shape[1])), dtype=np.float32)
    upper = np.asarray(getattr(problem, "xu", np.ones(archive_x_arr.shape[1])), dtype=np.float32)
    if lower.size == 1:
        lower = np.repeat(lower, int(archive_x_arr.shape[1]))
    if upper.size == 1:
        upper = np.repeat(upper, int(archive_x_arr.shape[1]))

    target_size = int(pop_size) if pop_size is not None else int(archive_x_arr.shape[0])
    if surrogate is None:
        surrogate = fit_gp_surrogates(
            archive_x=archive_x_arr,
            archive_y=archive_y_arr,
            seed=int(seed),
        )
    result = query_cdm_psl(
        archive_x=archive_x_arr,
        archive_y=archive_y_arr,
        xl=lower,
        xu=upper,
        surrogate=surrogate,
        n_select=int(n_select),
        use_diffusion=True,
        coef_lcb=float(coef_lcb),
        sbx_rounds=int(sbx_rounds),
        diffusion_augmentation_factor=int(diffusion_augmentation_factor),
        diffusion_num_steps=int(diffusion_num_steps),
        diffusion_batch_size=int(diffusion_batch_size),
        diffusion_num_epoch=int(diffusion_num_epoch),
        diffusion_guided_samples=int(diffusion_guided_samples),
        diffusion_random_samples=int(diffusion_random_samples),
        seed=int(seed),
    )
    offspring_x = np.asarray(result["offspring_x"], dtype=np.float32)
    offspring_mean = np.asarray(result["offspring_pred_mean"], dtype=np.float32)
    offspring_std = np.asarray(result["offspring_pred_std"], dtype=np.float32)
    keep = []
    keys = set()
    archive_ref = np.asarray(archive_x_arr, dtype=np.float32)
    for idx, row in enumerate(offspring_x):
        key = tuple(np.round(np.asarray(row, dtype=np.float64), decimals=10).tolist())
        if key in keys:
            continue
        if archive_ref.size and np.any(np.all(np.isclose(archive_ref, row[None, :], rtol=1e-8, atol=1e-8), axis=1)):
            continue
        keep.append(int(idx))
        keys.add(key)
        if len(keep) >= target_size:
            break
    if keep:
        idx = np.asarray(keep, dtype=np.int64)
        offspring_x = offspring_x[idx].copy()
        offspring_mean = offspring_mean[idx].copy()
        offspring_std = offspring_std[idx].copy()
    else:
        offspring_x = np.empty((0, int(archive_x_arr.shape[1])), dtype=np.float32)
        offspring_mean = np.empty((0, int(archive_y_arr.shape[1])), dtype=np.float32)
        offspring_std = np.empty((0, int(archive_y_arr.shape[1])), dtype=np.float32)
    if int(offspring_x.shape[0]) < target_size:
        rng = np.random.default_rng(int(seed) + 104729)
        selected = [row for row in offspring_x]
        selected_keys = {tuple(np.round(np.asarray(row, dtype=np.float64), decimals=10).tolist()) for row in selected}
        attempts = 0
        while len(selected) < target_size and attempts < 100 * target_size:
            attempts += 1
            row = rng.uniform(lower, upper, size=int(archive_x_arr.shape[1])).astype(np.float32)
            key = tuple(np.round(np.asarray(row, dtype=np.float64), decimals=10).tolist())
            if key in selected_keys:
                continue
            if archive_ref.size and np.any(np.all(np.isclose(archive_ref, row[None, :], rtol=1e-8, atol=1e-8), axis=1)):
                continue
            selected.append(row)
            selected_keys.add(key)
        random_x = np.asarray(selected[int(offspring_x.shape[0]):target_size], dtype=np.float32)
        if random_x.size:
            if hasattr(surrogate, "predict_mean"):
                random_mean = np.asarray(surrogate.predict_mean(random_x), dtype=np.float32)
            else:
                random_mean = np.asarray(surrogate.predict(random_x), dtype=np.float32)
            if hasattr(surrogate, "predict_std"):
                random_std = np.asarray(surrogate.predict_std(random_x), dtype=np.float32)
            else:
                _, random_std = surrogate.predict_mean_std(random_x)
                random_std = np.asarray(random_std, dtype=np.float32)
            offspring_x = np.vstack([offspring_x, random_x]).astype(np.float32)
            offspring_mean = np.vstack([offspring_mean, random_mean]).astype(np.float32)
            offspring_std = np.vstack([offspring_std, np.maximum(random_std, 1e-12)]).astype(np.float32)
    return offspring_x, offspring_mean, offspring_std


__all__ = ["CDMPSLQuerySolver", "query_cdm_psl", "run_surrogate_cdm_psl"]
