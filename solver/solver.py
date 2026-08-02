from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from pymoo.core.problem import Problem


@dataclass(frozen=True)
class SolverRequest:
    problem: Any
    archive_x: np.ndarray
    archive_y: np.ndarray | None = None
    surrogate: Any | None = None
    gps: Any | None = None
    pop_size: int = 80
    saea_steps: int = 25
    seed: int = 0
    n_mc: int = 128
    hybrid_nsga3_size: int | None = None
    acquisition: str = "ei"
    beta: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SolverResult:
    solver: str
    x: np.ndarray
    y: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    acquisition: np.ndarray | None = None
    raw: Any | None = None


SolverFn = Callable[[SolverRequest], SolverResult]


def _surrogate_predict_mean(surrogate, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_mean"):
        y = surrogate.predict_mean(x_arr)
    elif hasattr(surrogate, "predict"):
        y = surrogate.predict(x_arr)
    else:
        raise TypeError("surrogate must implement predict_mean(x) or predict(x).")
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim != 2:
        raise ValueError(f"surrogate prediction must return 2D (N, M), got shape={y_arr.shape}.")
    return y_arr


class GPSurrogateProblem(Problem):
    def __init__(self, surrogate, n_var, n_obj, xl, xu):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu)
        self.surrogate = surrogate

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = _surrogate_predict_mean(self.surrogate, np.asarray(X, dtype=np.float32))


class _ModelListSurrogate:
    def __init__(self, models):
        self.models = list(models)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        preds = []
        for model in self.models:
            if hasattr(model, "posterior"):
                model_device = next(model.parameters()).device
                x_tensor = torch.tensor(np.asarray(x_arr, dtype=np.float64), dtype=torch.double, device=model_device)
                with torch.no_grad():
                    mean = model.posterior(x_tensor).mean.detach().cpu().numpy().reshape(-1)
            elif hasattr(model, "predict"):
                mean = np.asarray(model.predict(x_arr), dtype=np.float32).reshape(-1)
            else:
                model_device = next(model.parameters()).device
                x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=model_device)
                with torch.no_grad():
                    mean = model(x_tensor).detach().cpu().numpy().reshape(-1)
            preds.append(np.asarray(mean, dtype=np.float32))
        return np.stack(preds, axis=1).astype(np.float32)


def normalize_solver_name(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    aliases = {
        "nsga_3": "nsga3",
        "moead_ego_solver": "moead_ego",
        "cdmpsl": "cdm_psl",
        "cdm_psl_solver": "cdm_psl",
        "hybrid_solver": "hybrid",
        "usemo_ei": "usemo",
    }
    return aliases.get(normalized, normalized)


def available_solvers() -> tuple[str, ...]:
    return tuple(sorted(_SOLVER_REGISTRY.keys()))


def get_solver(name: str) -> SolverFn:
    solver_name = normalize_solver_name(name)
    try:
        return _SOLVER_REGISTRY[solver_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported solver '{name}'. Available solvers: {', '.join(available_solvers())}."
        ) from exc


def solve(
    solver: str,
    *,
    problem,
    archive_x,
    archive_y=None,
    surrogate=None,
    gps=None,
    pop_size: int = 80,
    saea_steps: int = 25,
    seed: int = 0,
    n_mc: int = 128,
    hybrid_nsga3_size: int | None = None,
    acquisition: str = "ei",
    beta: float | None = None,
    surrogate_nsga_steps: int | None = None,
    **kwargs,
) -> SolverResult:
    if surrogate_nsga_steps is not None:
        saea_steps = surrogate_nsga_steps
    request = SolverRequest(
        problem=problem,
        archive_x=np.asarray(archive_x),
        archive_y=None if archive_y is None else np.asarray(archive_y),
        surrogate=surrogate,
        gps=gps,
        pop_size=int(pop_size),
        saea_steps=int(saea_steps),
        seed=int(seed),
        n_mc=int(n_mc),
        hybrid_nsga3_size=None if hybrid_nsga3_size is None else int(hybrid_nsga3_size),
        acquisition=str(acquisition),
        beta=beta,
        extra=dict(kwargs),
    )
    return get_solver(solver)(request)


def run_solver(solver: str, **kwargs) -> SolverResult:
    return solve(solver, **kwargs)


def _require_archive_y(request: SolverRequest, solver_name: str) -> np.ndarray:
    if request.archive_y is None:
        raise ValueError(f"solver '{solver_name}' requires archive_y.")
    return np.asarray(request.archive_y)


def _solve_nsga3(request: SolverRequest) -> SolverResult:
    from solver.nsga3_solver import run_surrogate_nsga3

    x, y = run_surrogate_nsga3(
        problem=request.problem,
        archive_x=request.archive_x,
        pop_size=int(request.pop_size),
        gps=request.gps,
        surrogate=request.surrogate,
        saea_steps=int(request.saea_steps),
        seed=int(request.seed),
        **request.extra,
    )
    return SolverResult(solver="nsga3", x=np.asarray(x), y=np.asarray(y), mean=np.asarray(y), raw=(x, y))


def _solve_moead_ego(request: SolverRequest) -> SolverResult:
    from solver.moead_ego_solver import run_surrogate_moead_ego

    archive_y = _require_archive_y(request, "moead_ego")
    if request.surrogate is None:
        raise ValueError("solver 'moead_ego' requires a surrogate object with predict_mean/predict_std.")
    result = run_surrogate_moead_ego(
        problem=request.problem,
        archive_x=request.archive_x,
        archive_y=archive_y,
        surrogate=request.surrogate,
        pop_size=int(request.pop_size),
        saea_steps=int(request.saea_steps),
        seed=int(request.seed),
        n_mc=int(request.n_mc),
        **request.extra,
    )
    return SolverResult(
        solver="moead_ego",
        x=np.asarray(result.x),
        y=np.asarray(result.mean),
        mean=np.asarray(result.mean),
        std=np.asarray(result.std),
        acquisition=np.asarray(result.eti),
        raw=result,
    )


def _solve_cdm_psl(request: SolverRequest) -> SolverResult:
    from baseline.cdm_psl_impl.cdm_psl_solver import run_surrogate_cdm_psl

    archive_y = _require_archive_y(request, "cdm_psl")
    x, mean, std = run_surrogate_cdm_psl(
        problem=request.problem,
        archive_x=request.archive_x,
        archive_y=archive_y,
        surrogate=request.surrogate,
        pop_size=int(request.pop_size),
        seed=int(request.seed),
        **request.extra,
    )
    return SolverResult(
        solver="cdm_psl",
        x=np.asarray(x),
        y=np.asarray(mean),
        mean=np.asarray(mean),
        std=np.asarray(std),
        raw=(x, mean, std),
    )


def _solve_hybrid(request: SolverRequest) -> SolverResult:
    from solver.hybrid_solver import run_surrogate_hybrid

    archive_y = _require_archive_y(request, "hybrid")
    if request.surrogate is None:
        raise ValueError("solver 'hybrid' requires a fitted surrogate object.")
    result = run_surrogate_hybrid(
        problem=request.problem,
        archive_x=request.archive_x,
        archive_y=archive_y,
        surrogate=request.surrogate,
        pop_size=int(request.pop_size),
        saea_steps=int(request.saea_steps),
        seed=int(request.seed),
        n_mc=int(request.n_mc),
        nsga3_pop_size=request.hybrid_nsga3_size,
        **request.extra,
    )
    return SolverResult(
        solver="hybrid",
        x=np.asarray(result.x),
        y=np.asarray(result.mean),
        mean=np.asarray(result.mean),
        std=np.asarray(result.std),
        raw=result,
    )


def _solve_usemo(request: SolverRequest) -> SolverResult:
    from solver.usemo_solver import run_surrogate_usemo

    archive_y = _require_archive_y(request, "usemo")
    x, mean, std = run_surrogate_usemo(
        problem=request.problem,
        archive_x=request.archive_x,
        archive_y=archive_y,
        pop_size=int(request.pop_size),
        saea_steps=int(request.saea_steps),
        seed=int(request.seed),
        acquisition=str(request.acquisition),
        beta=request.beta,
        **request.extra,
    )
    return SolverResult(
        solver="usemo",
        x=np.asarray(x),
        y=np.asarray(mean),
        mean=np.asarray(mean),
        std=np.asarray(std),
        raw=(x, mean, std),
    )


_SOLVER_REGISTRY: dict[str, SolverFn] = {
    "cdm_psl": _solve_cdm_psl,
    "hybrid": _solve_hybrid,
    "moead_ego": _solve_moead_ego,
    "nsga3": _solve_nsga3,
    "usemo": _solve_usemo,
}


__all__ = [
    "GPSurrogateProblem",
    "SolverRequest",
    "SolverResult",
    "_ModelListSurrogate",
    "available_solvers",
    "get_solver",
    "normalize_solver_name",
    "run_solver",
    "solve",
]
