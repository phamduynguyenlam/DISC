from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.stats import norm

from surrogate.surrogate_model import fit_gp_surrogates

USEMO_GP_CONFIG: dict[str, object] = {
    "normalize_x": False,
    "standardize_y": False,
    "winsorize_y_quantile": None,
    "kernel": "rbf",
    "nu": 0.0,
    "ard": False,
    "output_model": "independent_gp_per_objective",
    "likelihood": "gaussian",
    "noise_constraint": [1e-5, 5e-2],
    "initial_noise": 1e-4,
    "lengthscale_constraint": [0.05, 2.0],
    "initial_lengthscale": 0.30,
    "outputscale_constraint": [0.05, 20.0],
    "max_fit_iter": 80,
    "num_restarts": 4,
    "jitter": 1e-6,
    "surrogate_objective": "mean",
    "cheap_solver": "NSGA-II",
    "saea_steps": 30,
    "pop_size": 80,
    "n_restarts": 1,
    "candidate_pool_size": 80,
    "keep_top_k": 80,
    "filter_invalid": False,
}


@dataclass(frozen=True)
class USEMOResult:
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray


class _ConfiguredUSEMOGP:
    def __init__(self, *, xl: np.ndarray, xu: np.ndarray):
        self.xl = np.asarray(xl, dtype=np.float64).reshape(1, -1)
        self.xu = np.asarray(xu, dtype=np.float64).reshape(1, -1)
        self.x_span = np.clip(self.xu - self.xl, 1e-12, None)
        self.y_low: np.ndarray | None = None
        self.y_high: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.model = None

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if bool(USEMO_GP_CONFIG["normalize_x"]):
            return np.clip((x_arr - self.xl) / self.x_span, 0.0, 1.0)
        return x_arr

    def _winsorize_y(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        q_value = USEMO_GP_CONFIG["winsorize_y_quantile"]
        if q_value is None:
            self.y_low = np.min(y_arr, axis=0)
            self.y_high = np.max(y_arr, axis=0)
            return y_arr
        q = float(q_value)
        if q <= 0.0:
            self.y_low = np.min(y_arr, axis=0)
            self.y_high = np.max(y_arr, axis=0)
            return y_arr
        self.y_low = np.quantile(y_arr, q, axis=0)
        self.y_high = np.quantile(y_arr, 1.0 - q, axis=0)
        return np.clip(y_arr, self.y_low, self.y_high)

    def _standardize_y(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        if bool(USEMO_GP_CONFIG["standardize_y"]):
            self.y_mean = np.mean(y_arr, axis=0)
            self.y_scale = np.std(y_arr, axis=0)
            self.y_scale = np.where(self.y_scale < 1e-8, 1.0, self.y_scale)
            return (y_arr - self.y_mean) / self.y_scale
        self.y_mean = np.zeros(y_arr.shape[1], dtype=np.float64)
        self.y_scale = np.ones(y_arr.shape[1], dtype=np.float64)
        return y_arr

    def fit(self, archive_x: np.ndarray, archive_y: np.ndarray, *, seed: int) -> "_ConfiguredUSEMOGP":
        x_arr = np.asarray(archive_x, dtype=np.float64)
        y_arr = np.asarray(archive_y, dtype=np.float64)
        self.y_low = np.min(y_arr, axis=0)
        self.y_high = np.max(y_arr, axis=0)
        self.y_mean = np.zeros(y_arr.shape[1], dtype=np.float64)
        self.y_scale = np.ones(y_arr.shape[1], dtype=np.float64)
        self.model = fit_gp_surrogates(
            archive_x=x_arr,
            archive_y=y_arr,
            seed=int(seed),
            nu=float(USEMO_GP_CONFIG["nu"]),
        )
        return self

    def _restore_mean(self, mean: np.ndarray) -> np.ndarray:
        mean_arr = np.asarray(mean, dtype=np.float64)
        assert self.y_mean is not None
        assert self.y_scale is not None
        restored = mean_arr * self.y_scale.reshape(1, -1) + self.y_mean.reshape(1, -1)
        return restored.astype(np.float32)

    def _restore_std(self, std: np.ndarray) -> np.ndarray:
        std_arr = np.asarray(std, dtype=np.float64)
        assert self.y_scale is not None
        restored = std_arr * self.y_scale.reshape(1, -1)
        return np.maximum(restored, 1e-6).astype(np.float32)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x_arr = self._normalize_x(np.asarray(x, dtype=np.float64))
        return np.asarray(self.model.predict_mean(x_arr), dtype=np.float32)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        x_arr = self._normalize_x(np.asarray(x, dtype=np.float64))
        return np.asarray(self.model.predict_std(x_arr), dtype=np.float32)

    def incumbent(self) -> np.ndarray:
        assert self.y_low is not None
        assert self.y_high is not None
        assert self.y_mean is not None
        assert self.y_scale is not None
        y_ref = np.clip(self.y_mean.reshape(1, -1), self.y_low.reshape(1, -1), self.y_high.reshape(1, -1))
        return y_ref.reshape(-1).astype(np.float32)


def _pareto_mask(values: np.ndarray) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    if values_arr.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values_arr.shape}.")
    keep = np.zeros(int(values_arr.shape[0]), dtype=bool)
    indices = NonDominatedSorting().do(values_arr, only_non_dominated_front=True)
    keep[np.asarray(indices, dtype=np.int64)] = True
    return keep


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    incumbent: np.ndarray,
) -> np.ndarray:
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32).clip(min=1e-12)
    incumbent_arr = np.asarray(incumbent, dtype=np.float32).reshape(1, -1)
    improvement = incumbent_arr - mean_arr
    z = improvement / std_arr
    ei = improvement * norm.cdf(z) + std_arr * norm.pdf(z)
    return np.asarray(ei, dtype=np.float32)


class _USEMOProblem(Problem):
    def __init__(
        self,
        gp_suite,
        incumbent: np.ndarray,
        archive_y: np.ndarray,
        n_var: int,
        n_obj: int,
        xl: np.ndarray,
        xu: np.ndarray,
    ):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.gp_suite = gp_suite
        self.incumbent = np.asarray(incumbent, dtype=np.float32)
        self.archive_y = np.asarray(archive_y, dtype=np.float32)

    def _evaluate(self, X, out, *args, **kwargs):
        x_arr = np.asarray(X, dtype=np.float32)
        mean = np.asarray(self.gp_suite.predict_mean(x_arr), dtype=np.float32)
        std = np.asarray(self.gp_suite.predict_std(x_arr), dtype=np.float32)
        ei = _expected_improvement(mean, std, self.incumbent)
        out["F"] = (-ei).astype(np.float32)


def _filter_valid_candidates(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    valid = np.all(np.isfinite(x), axis=1)
    valid &= np.all(np.isfinite(mean), axis=1)
    valid &= np.all(np.isfinite(std), axis=1)
    valid &= np.all(std > 0.0, axis=1)
    return valid


def _filter_not_evaluated_candidates(
    x: np.ndarray,
    archive_x: np.ndarray,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    archive_arr = np.asarray(archive_x, dtype=np.float64)
    if x_arr.size == 0:
        return np.zeros(0, dtype=bool)
    if archive_arr.size == 0:
        return np.ones(int(x_arr.shape[0]), dtype=bool)
    deltas = x_arr[:, None, :] - archive_arr[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return np.all(distances >= float(tol), axis=1)


def run_surrogate_usemo(
    problem,
    archive_x,
    archive_y,
    pop_size,
    saea_steps=30,
    seed=0,
    n_gen=None,
    surrogate_nsga_steps=None,
):
    if surrogate_nsga_steps is not None:
        saea_steps = surrogate_nsga_steps
    if n_gen is not None:
        saea_steps = n_gen
    if saea_steps is None:
        saea_steps = int(USEMO_GP_CONFIG["saea_steps"])

    archive_x_arr = np.asarray(archive_x, dtype=np.float64)
    archive_y_arr = np.asarray(archive_y, dtype=np.float64)
    xl = np.asarray(problem.xl, dtype=np.float32)
    xu = np.asarray(problem.xu, dtype=np.float32)
    gp_suite = _ConfiguredUSEMOGP(xl=xl, xu=xu).fit(
        archive_x=archive_x_arr,
        archive_y=archive_y_arr,
        seed=int(seed),
    )

    surrogate_problem = _USEMOProblem(
        gp_suite=gp_suite,
        incumbent=np.min(np.asarray(archive_y_arr, dtype=np.float32), axis=0),
        archive_y=np.asarray(archive_y_arr, dtype=np.float32),
        n_var=int(problem.n_var),
        n_obj=int(problem.n_obj),
        xl=xl,
        xu=xu,
    )

    all_x: list[np.ndarray] = []
    all_acq: list[np.ndarray] = []
    n_restarts = int(USEMO_GP_CONFIG["n_restarts"])
    for restart_id in range(n_restarts):
        algorithm = NSGA2(
            pop_size=int(pop_size),
            sampling=LHS(),
            eliminate_duplicates=True,
        )
        res = pymoo_minimize(
            surrogate_problem,
            algorithm,
            termination=get_termination("n_gen", int(saea_steps)),
            seed=int(seed) + int(restart_id),
            verbose=False,
            save_history=False,
        )
        final_pop = res.pop
        final_x = np.asarray(final_pop.get("X"), dtype=np.float32)
        final_f = np.asarray(final_pop.get("F"), dtype=np.float32)
        if final_x.ndim == 1:
            final_x = final_x.reshape(1, -1)
        if final_f.ndim == 1:
            final_f = final_f.reshape(1, -1)
        all_x.append(final_x)
        all_acq.append(final_f)

    x_res = np.vstack(all_x).astype(np.float32)
    acq_res = np.vstack(all_acq).astype(np.float32)
    if int(USEMO_GP_CONFIG["candidate_pool_size"]) > 0 and x_res.shape[0] > int(USEMO_GP_CONFIG["candidate_pool_size"]):
        x_res = x_res[: int(USEMO_GP_CONFIG["candidate_pool_size"])].copy()
        acq_res = acq_res[: int(USEMO_GP_CONFIG["candidate_pool_size"])].copy()

    mean_res = np.asarray(gp_suite.predict_mean(x_res), dtype=np.float32)
    std_res = np.asarray(gp_suite.predict_std(x_res), dtype=np.float32)
    if bool(USEMO_GP_CONFIG["filter_invalid"]):
        valid_mask = _filter_valid_candidates(x_res, mean_res, std_res)
        x_res = x_res[valid_mask]
        acq_res = acq_res[valid_mask]
        mean_res = mean_res[valid_mask]
        std_res = std_res[valid_mask]

    mask = _pareto_mask(acq_res)
    x_res = np.asarray(x_res[mask], dtype=np.float32)
    mean_res = np.asarray(mean_res[mask], dtype=np.float32)
    std_res = np.asarray(std_res[mask], dtype=np.float32)

    not_evaluated_mask = _filter_not_evaluated_candidates(x_res, archive_x_arr)
    x_res = np.asarray(x_res[not_evaluated_mask], dtype=np.float32)
    mean_res = np.asarray(mean_res[not_evaluated_mask], dtype=np.float32)
    std_res = np.asarray(std_res[not_evaluated_mask], dtype=np.float32)

    return (
        x_res,
        mean_res,
        std_res,
    )
