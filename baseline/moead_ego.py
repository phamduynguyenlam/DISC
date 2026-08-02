"""
MOEA/D-EGO baseline for expensive multi-objective optimization.

Setting used by default:
    - 80 initial true evaluations (Latin hypercube)
    - 40 additional true evaluations (one infill point per iteration)
    - total FE = 120

This is a compact Python implementation inspired by the public MATLAB/PlatEMO
MOEA-D-EGO implementation from https://github.com/mobo-d/MOEAD-EGO.
The original code maximizes Expected Tchebycheff Improvement (ETI) by MOEA/D-DE.
Here ETI is estimated by Monte Carlo under independent GP posteriors, which is
slower than the exact formula but much simpler and portable.

Dependencies:
    numpy scipy scikit-learn
Optional for CLI examples:
    pymoo

Example:
    python moead_ego.py --problem zdt1 --n-var 30 --seed 0 \
        --init-samples 80 --infill-budget 40 --moead-pop 200 --saea_steps 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.cluster import KMeans
from pymoo.operators.mutation.pm import mut_pm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lhs import latin_hypercube_sample
from problem.problem import get_reference_point
from reward import hypervolume
from surrogate.surrogate_model import fit_gp_surrogates

Array = np.ndarray
PLOT_WIDTH_4_3 = 8.0
PLOT_HEIGHT_4_3 = 6.0
WIDE_PLOT_WIDTH_4_3 = 12.0
WIDE_PLOT_HEIGHT_4_3 = 9.0


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = log_path.open("a", encoding="utf-8")

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        fp.write(text + "\n")
        fp.flush()

    return _log, fp


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def default_log_dir(args: argparse.Namespace) -> Path:
    return repo_path("testing_logs", str(args.problem).upper())


def default_log_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"moead_ego_{str(args.problem).lower()}_seed{int(args.seed)}_{timestamp}.txt"
    return default_log_dir(args) / stem


def default_plot_dir(args: argparse.Namespace) -> Path:
    base_dir = repo_path("png")
    return base_dir / str(args.problem).upper()


def resolve_pdf_plot_path(plot_file: Path, args: argparse.Namespace) -> Path:
    base_dir = repo_path("pdf")
    return base_dir / str(args.problem).upper() / f"{plot_file.stem}.pdf"


def save_plot_outputs(fig, *, plot_file: Path) -> Path:
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=180)
    return plot_file.resolve()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def latin_hypercube(n: int, d: int, lower: Array, upper: Array, seed: Optional[int]) -> Array:
    return latin_hypercube_sample(
        n_samples=int(n),
        dim=int(d),
        lower=np.asarray(lower, dtype=np.float32),
        upper=np.asarray(upper, dtype=np.float32),
        seed=0 if seed is None else int(seed),
    ).astype(float)


def normalize_x(x: Array, lower: Array, upper: Array) -> Array:
    return (x - lower) / np.maximum(upper - lower, 1e-12)


def denormalize_x(xn: Array, lower: Array, upper: Array) -> Array:
    return lower + xn * (upper - lower)


def non_dominated_mask(y: Array) -> Array:
    """Return True for nondominated rows, assuming minimization."""
    keep = np.zeros(int(y.shape[0]), dtype=bool)
    indices = NonDominatedSorting().do(y, only_non_dominated_front=True)
    keep[np.asarray(indices, dtype=np.int64)] = True
    return keep


def pareto_front(y: Array) -> Array:
    y_arr = np.asarray(y, dtype=float)
    return y_arr[non_dominated_mask(y_arr)]


def generate_weights(m: int, n_weights: int) -> Array:
    """Generate an exact-size set of simplex directions with pymoo."""
    if int(m) <= 1:
        return np.ones((int(n_weights), 1), dtype=float)
    try:
        weights = get_reference_directions("energy", int(m), int(n_weights), seed=0)
    except TypeError:
        weights = get_reference_directions("energy", int(m), int(n_weights))
    return np.maximum(weights, 1e-6)


def tchebycheff(y: Array, weights: Array, z: Array) -> Array:
    """
    Compute g(x | w, z) = max_j w_j * (f_j(x) - z_j).
    y:       (n, m)
    weights: (k, m)
    returns: (k, n)
    """
    return np.max(weights[:, None, :] * (y[None, :, :] - z[None, None, :]), axis=2)


def de_polynomial_mutation(x: Array, lower: Array, upper: Array, rng: np.random.Generator,
                           eta: float = 20.0, prob: Optional[float] = None) -> Array:
    y = np.asarray(x, dtype=float).reshape(1, -1)
    d = int(y.shape[1])
    if prob is None:
        prob = 1.0 / d
    return mut_pm(
        y,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        eta=np.full(1, float(eta)),
        prob=np.full(1, float(prob)),
        at_least_once=False,
        random_state=rng,
    )[0]


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------

@dataclass
class RepoGPSurrogate:
    model: object

    @staticmethod
    def fit(x: Array, y: Array, seed: int = 0, n_restarts: int = 3) -> "RepoGPSurrogate":
        del n_restarts
        model = fit_gp_surrogates(
            archive_x=np.asarray(x, dtype=np.float32),
            archive_y=np.asarray(y, dtype=np.float32),
            seed=int(seed),
        )
        return RepoGPSurrogate(model=model)

    def predict(self, x: Array) -> Tuple[Array, Array]:
        x_arr = np.atleast_2d(np.asarray(x, dtype=np.float32))
        mean = np.asarray(self.model.predict_mean(x_arr), dtype=np.float32)
        std = np.asarray(self.model.predict_std(x_arr), dtype=np.float32)
        return mean, np.maximum(std, 1e-12)


# ---------------------------------------------------------------------------
# MOEA/D-EGO inner optimizer
# ---------------------------------------------------------------------------

@dataclass
class MOEADEGOConfig:
    init_samples: int = 80
    infill_budget: int = 40
    batch_size: int = 5
    moead_pop: int = 200
    saea_steps: int = 30
    neighbor_frac: float = 0.1
    delta: float = 0.9
    nr: int = 2
    de_f: float = 0.5
    de_cr: float = 1.0
    gp_restarts: int = 3
    duplicate_tol: float = 1e-6
    seed: int = 0
    ref_point: Array | None = None


class MOEADEGOSolver:
    """
    Minimization-only MOEA/D-EGO baseline.

    objective must accept x with shape (n, d) and return y with shape (n, m).
    """

    def __init__(
        self,
        objective: Callable[[Array], Array],
        lower: Array,
        upper: Array,
        n_obj: int,
        config: Optional[MOEADEGOConfig] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.objective = objective
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.n_var = len(self.lower)
        self.n_obj = n_obj
        self.cfg = config or MOEADEGOConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.logger = logger

    def _max_two_gaussians(self, mu1: Array, sigma1: Array, mu2: Array, sigma2: Array) -> Tuple[Array, Array]:
        mu1_arr = np.asarray(mu1, dtype=float)
        mu2_arr = np.asarray(mu2, dtype=float)
        sigma1_arr = np.maximum(np.asarray(sigma1, dtype=float), 1e-12)
        sigma2_arr = np.maximum(np.asarray(sigma2, dtype=float), 1e-12)
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
        return np.asarray(mean, dtype=float), np.asarray(np.sqrt(var), dtype=float)

    def _tchebycheff_gaussian_stats(self, mu: Array, std: Array, weights: Array, z: Array) -> Tuple[Array, Array]:
        mu_arr = np.asarray(mu, dtype=float)
        std_arr = np.maximum(np.asarray(std, dtype=float), 1e-12)
        weights_arr = np.asarray(weights, dtype=float)
        z_arr = np.asarray(z, dtype=float).reshape(1, -1)
        weighted_mean = weights_arr * (mu_arr - z_arr)
        weighted_std = weights_arr * std_arr
        if weighted_mean.shape[1] == 1:
            return weighted_mean[:, 0], np.maximum(weighted_std[:, 0], 1e-12)

        agg_mean = weighted_mean[:, 0]
        agg_std = np.maximum(weighted_std[:, 0], 1e-12)
        for obj_idx in range(1, weighted_mean.shape[1]):
            agg_mean, agg_std = self._max_two_gaussians(
                agg_mean,
                agg_std,
                weighted_mean[:, obj_idx],
                weighted_std[:, obj_idx],
            )
        return np.asarray(agg_mean, dtype=float), np.asarray(np.maximum(agg_std, 1e-12), dtype=float)

    def _expected_tchebycheff_improvement(self, mu: Array, std: Array, weights: Array,
                                          gmin: Array, z: Array) -> Array:
        g_mean, g_std = self._tchebycheff_gaussian_stats(mu, std, weights, z)
        gmin_arr = np.asarray(gmin, dtype=float).reshape(-1)
        g_mean_arr = np.asarray(g_mean, dtype=float).reshape(-1)
        g_std_arr = np.maximum(np.asarray(g_std, dtype=float).reshape(-1), 1e-12)
        u = (gmin_arr - g_mean_arr) / g_std_arr
        ei = (gmin_arr - g_mean_arr) * norm.cdf(u) + g_std_arr * norm.pdf(u)
        return np.asarray(np.maximum(ei, 0.0), dtype=float)

    def _optimize_eti(self, gp: RepoGPSurrogate, x_train: Array, y_train: Array) -> Tuple[Array, Dict[str, float]]:
        cfg = self.cfg
        weights = generate_weights(self.n_obj, cfg.moead_pop)
        pop_size = len(weights)

        # Utopian point. The MATLAB repo estimates z adaptively; this conservative
        # variant uses best observed objective value minus a small margin.
        z = np.min(y_train, axis=0) - 1e-6 * np.maximum(np.std(y_train, axis=0), 1.0)
        gmin = np.min(tchebycheff(y_train, weights, z), axis=1)

        # Neighborhood by weight-vector distance.
        t = max(2, int(np.ceil(cfg.neighbor_frac * pop_size)))
        neigh = np.argsort(cdist(weights, weights), axis=1)[:, :t]

        # Initial candidate population in decision space.
        pop_x = latin_hypercube(pop_size, self.n_var, self.lower, self.upper,
                                seed=int(self.rng.integers(1_000_000_000)))
        mu, std = gp.predict(pop_x)
        pop_eti = self._expected_tchebycheff_improvement(mu, std, weights, gmin, z)

        for _ in range(cfg.saea_steps):
            for i in range(pop_size):
                if self.rng.random() < cfg.delta:
                    pool = neigh[i]
                else:
                    pool = np.arange(pop_size)
                p = self.rng.choice(pool, size=2, replace=False)

                trial = pop_x[i].copy()
                mask = self.rng.random(self.n_var) < cfg.de_cr
                trial[mask] = pop_x[i, mask] + cfg.de_f * (pop_x[p[0], mask] - pop_x[p[1], mask])
                trial = np.clip(trial, self.lower, self.upper)
                trial = de_polynomial_mutation(trial, self.lower, self.upper, self.rng)

                off_mu, off_std = gp.predict(trial[None, :])
                cand_weights = weights[pool]
                cand_gmin = gmin[pool]
                off_mu_rep = np.repeat(off_mu, len(pool), axis=0)
                off_std_rep = np.repeat(off_std, len(pool), axis=0)
                off_eti = self._expected_tchebycheff_improvement(off_mu_rep, off_std_rep, cand_weights, cand_gmin, z)

                improve_local = np.where(off_eti > pop_eti[pool])[0]
                if len(improve_local) > 0:
                    # Replace at most nr neighbors, as in MOEA/D restricted update.
                    chosen = improve_local[: cfg.nr]
                    idx = pool[chosen]
                    pop_x[idx] = trial
                    pop_eti[idx] = off_eti[chosen]

        # Filter duplicate / already evaluated points and keep positive ETI first.
        dist_to_train = cdist(pop_x, x_train).min(axis=1)
        valid = dist_to_train > cfg.duplicate_tol
        if np.any(pop_eti > 0):
            valid &= pop_eti > 0

        cand_x = pop_x[valid]
        cand_eti = pop_eti[valid]
        cand_weights = weights[valid]
        if len(cand_x) == 0:
            # Fallback: random unevaluated point.
            x = latin_hypercube(cfg.batch_size, self.n_var, self.lower, self.upper,
                                seed=int(self.rng.integers(1_000_000_000)))
            return x, {"best_eti": 0.0, "mean_eti": 0.0, "n_candidates": 0}

        if cfg.batch_size == 1:
            idx = int(np.argmax(cand_eti))
            selected = cand_x[[idx]]
        else:
            # Batch selection: cluster candidate weight vectors, then pick best EI in each cluster.
            b = min(cfg.batch_size, len(cand_x))
            labels = KMeans(n_clusters=b, n_init=10, random_state=cfg.seed).fit_predict(cand_weights)
            selected_idx = []
            for k in range(b):
                members = np.where(labels == k)[0]
                selected_idx.append(members[np.argmax(cand_eti[members])])
            selected = cand_x[selected_idx]

        return selected, {
            "best_eti": float(np.max(cand_eti)),
            "mean_eti": float(np.mean(cand_eti)),
            "n_candidates": int(len(cand_x)),
        }

    def run(self) -> Dict[str, Array | List[Dict[str, float]]]:
        cfg = self.cfg
        x = latin_hypercube(cfg.init_samples, self.n_var, self.lower, self.upper, cfg.seed)
        y = np.asarray(self.objective(x), dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        logs: List[Dict[str, float]] = []
        init_nd = non_dominated_mask(y)
        init_hv = float(hypervolume(y, cfg.ref_point)) if cfg.ref_point is not None else float("nan")
        init_log = {
            "iter": 0,
            "fe": int(len(x)),
            "archive_size": int(len(x)),
            "n_nd": int(init_nd.sum()),
            "hv": init_hv,
            "best_eti": float("nan"),
            "mean_eti": float("nan"),
            "n_candidates": 0,
        }
        logs.append(init_log)
        init_hv_part = f" | HV = {init_log['hv']:.12f}" if cfg.ref_point is not None else " | HV = nan"
        init_message = f"[moead_ego] iter 0 | front = {init_log['n_nd']}{init_hv_part}"
        if self.logger is not None:
            self.logger(init_message)
        else:
            print(init_message)

        fe = len(x)
        remaining = cfg.infill_budget
        iteration = 0
        while remaining > 0:
            batch = min(cfg.batch_size, remaining)
            old_batch = cfg.batch_size
            cfg.batch_size = batch

            gp = RepoGPSurrogate.fit(x, y, seed=cfg.seed, n_restarts=cfg.gp_restarts)
            new_x, info = self._optimize_eti(gp, x, y)
            new_y = np.asarray(self.objective(new_x), dtype=float)
            if new_y.ndim == 1:
                new_y = new_y[:, None]
            cfg.batch_size = old_batch
            for local_idx in range(int(new_x.shape[0])):
                selected_x = np.asarray(new_x[local_idx : local_idx + 1], dtype=float)
                selected_y = np.asarray(new_y[local_idx : local_idx + 1], dtype=float)
                x = np.vstack([x, selected_x])
                y = np.vstack([y, selected_y])
                fe += 1
                remaining -= 1

                nd = non_dominated_mask(y)
                log = {
                    "iter": iteration + 1,
                    "fe": fe,
                    "archive_size": len(x),
                    "n_nd": int(nd.sum()),
                    "hv": float(hypervolume(y, cfg.ref_point)) if cfg.ref_point is not None else float("nan"),
                    "best_eti": info["best_eti"],
                    "mean_eti": info["mean_eti"],
                    "n_candidates": info["n_candidates"],
                }
                logs.append(log)
                hv_part = f" | HV = {log['hv']:.12f}" if cfg.ref_point is not None else " | HV = nan"
                message = f"[moead_ego] iter {iteration + 1} | front = {log['n_nd']}{hv_part}"
                if self.logger is not None:
                    self.logger(message)
                else:
                    print(message)
                iteration += 1

        nd_mask = non_dominated_mask(y)
        return {
            "X": x,
            "Y": y,
            "nd_mask": nd_mask,
            "pareto_X": x[nd_mask],
            "pareto_Y": y[nd_mask],
            "logs": logs,
        }


def plot_results(*, args: argparse.Namespace, archive_y: Array, fe_history: list[int], hv_history: list[float]) -> str | None:
    archive_y_arr = np.asarray(archive_y, dtype=float)
    if archive_y_arr.ndim != 2 or archive_y_arr.shape[0] <= 0:
        return None
    n_obj = int(archive_y_arr.shape[1])
    if n_obj not in (2, 3):
        return None

    front = pareto_front(archive_y_arr)
    fig = plt.figure(figsize=(WIDE_PLOT_WIDTH_4_3, WIDE_PLOT_HEIGHT_4_3))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4, color="#9467bd")
    ax_hv.set_title("Hypervolume")
    ax_hv.set_xlabel("Function Evaluations")
    ax_hv.set_ylabel("HV")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y_arr[:, 0], archive_y_arr[:, 1], s=18, alpha=0.35, label="Archive", color="#c5b0d5")
        ax_pf.scatter(front[:, 0], front[:, 1], s=28, label="Final Front", color="#9467bd")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
    else:
        ax_pf.scatter(archive_y_arr[:, 0], archive_y_arr[:, 1], archive_y_arr[:, 2], s=18, alpha=0.35, label="Archive", color="#c5b0d5")
        ax_pf.scatter(front[:, 0], front[:, 1], front[:, 2], s=28, label="Final Front", color="#9467bd")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    ax_pf.set_title("Final Pareto Front")
    ax_pf.legend(loc="best")

    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return None


def problem_bounds(problem, archive_x: Array) -> Tuple[Array, Array]:
    x_arr = np.asarray(archive_x, dtype=float)
    dim = int(x_arr.shape[1])
    lower = np.asarray(getattr(problem, "xl", np.zeros(dim)), dtype=float).reshape(-1)
    upper = np.asarray(getattr(problem, "xu", np.ones(dim)), dtype=float).reshape(-1)
    if lower.size == 1:
        lower = np.repeat(lower, dim)
    if upper.size == 1:
        upper = np.repeat(upper, dim)
    return lower.astype(float), upper.astype(float)


def propose_moead_ego_candidates(
    *,
    problem,
    archive_x: Array,
    archive_y: Array,
    pop_size: int | None = None,
    saea_steps: int = 30,
    batch_size: int = 1,
    gp_restarts: int = 3,
    seed: int = 0,
) -> Tuple[Array, Array, Array, Dict[str, float]]:
    """Propose MOEA/D-EGO candidates from the current archive.

    This adapter is intended for tester.py compare flows. It reuses the archive
    supplied by the tester, fits the internal GP, optimizes ETI, and returns
    predicted mean/std for the proposed candidates without evaluating them.
    """
    archive_x_arr = np.asarray(archive_x, dtype=float)
    archive_y_arr = np.asarray(archive_y, dtype=float)
    if archive_x_arr.ndim != 2 or archive_y_arr.ndim != 2:
        raise ValueError("archive_x and archive_y must be 2D arrays.")
    if archive_x_arr.shape[0] != archive_y_arr.shape[0]:
        raise ValueError(
            f"archive_x/archive_y row count mismatch: {archive_x_arr.shape[0]} vs {archive_y_arr.shape[0]}."
        )

    lower, upper = problem_bounds(problem, archive_x_arr)
    cfg = MOEADEGOConfig(
        init_samples=int(archive_x_arr.shape[0]),
        infill_budget=int(batch_size),
        batch_size=int(batch_size),
        moead_pop=int(archive_x_arr.shape[0]) if pop_size is None else int(pop_size),
        saea_steps=int(saea_steps),
        gp_restarts=int(gp_restarts),
        seed=int(seed),
    )

    def _unused_objective(x: Array) -> Array:
        return np.zeros((np.atleast_2d(x).shape[0], archive_y_arr.shape[1]), dtype=float)

    solver = MOEADEGOSolver(
        objective=_unused_objective,
        lower=lower,
        upper=upper,
        n_obj=int(archive_y_arr.shape[1]),
        config=cfg,
        logger=None,
    )
    gp = RepoGPSurrogate.fit(
        archive_x_arr,
        archive_y_arr,
        seed=int(seed),
        n_restarts=int(gp_restarts),
    )
    candidate_x, info = solver._optimize_eti(gp, archive_x_arr, archive_y_arr)
    candidate_mean, candidate_std = gp.predict(candidate_x)
    info = dict(info)
    info["reference_vectors"] = generate_weights(int(archive_y_arr.shape[1]), int(cfg.moead_pop)).astype(np.float32)
    return (
        np.asarray(candidate_x, dtype=np.float32),
        np.asarray(candidate_mean, dtype=np.float32),
        np.asarray(candidate_std, dtype=np.float32),
        info,
    )


# ---------------------------------------------------------------------------
# CLI adapter for pymoo problems
# ---------------------------------------------------------------------------

def make_pymoo_objective(problem_name: str, n_var: int):
    try:
        from pymoo.problems import get_problem

        problem = get_problem(problem_name, n_var=n_var)
        lower = np.asarray(problem.xl, dtype=float)
        upper = np.asarray(problem.xu, dtype=float)
        n_obj = int(problem.n_obj)

        def obj(x: Array) -> Array:
            return problem.evaluate(np.atleast_2d(x), return_values_of=["F"])

        return obj, lower, upper, n_obj
    except Exception:
        from problem.problem import make_problem

        problem = make_problem(problem_name, dim=int(n_var))
        lower = np.asarray(problem.lower, dtype=float).reshape(-1)
        upper = np.asarray(problem.upper, dtype=float).reshape(-1)
        if lower.size == 1:
            lower = np.repeat(lower, int(problem.dim))
        if upper.size == 1:
            upper = np.repeat(upper, int(problem.dim))
        sample_y = np.asarray(problem.evaluate(np.atleast_2d(lower)), dtype=np.float32)
        n_obj = int(sample_y.shape[1])

    def obj(x: Array) -> Array:
        return np.asarray(problem.evaluate(np.atleast_2d(x)), dtype=np.float32)

    return obj, lower, upper, n_obj


def run_once(run_args: argparse.Namespace) -> dict:
    run_started_at = time.perf_counter()
    log_path = Path(run_args.log_path) if run_args.log_path else default_log_path(run_args)
    logger, log_fp = make_logger(log_path)
    try:
        logger(f"test_log_path = {log_path.resolve()}")
        logger(f"dim = {int(run_args.dim)}")
        logger("framework_label = MOEA/D-EGO")
        logger("infill_label = MOEA/D-EGO")
        logger("comparison_family = solver_vs_baseline")
        logger("solver_family = moead_ego")
        logger("run_variant = baseline")
        objective, lower, upper, n_obj = make_pymoo_objective(run_args.problem, run_args.dim)
        try:
            ref_point = get_reference_point(str(run_args.problem), n_obj=int(n_obj))
        except Exception:
            ref_point = None
        cfg = MOEADEGOConfig(
            init_samples=run_args.init_samples,
            infill_budget=run_args.infill_budget,
            batch_size=run_args.batch_size,
            moead_pop=run_args.moead_pop,
            saea_steps=run_args.saea_steps,
            gp_restarts=run_args.gp_restarts,
            seed=run_args.seed,
            ref_point=ref_point,
        )
        solver = MOEADEGOSolver(objective, lower, upper, n_obj, cfg, logger=logger)
        result = solver.run()

        fe_history = [int(log["fe"]) for log in result["logs"]]
        if ref_point is not None:
            hv_history = [float(log["hv"]) for log in result["logs"]]
        else:
            hv_history = [float("nan")] * len(fe_history)

        summary = {
            "method": "moead_ego",
            "problem": str(run_args.problem),
            "dim": int(run_args.dim),
            "seed": int(run_args.seed),
            "init_fe": int(run_args.init_samples),
            "max_fe": int(run_args.init_samples) + int(run_args.infill_budget),
            "batch_size": int(run_args.batch_size),
            "moead_pop": int(run_args.moead_pop),
            "saea_steps": int(run_args.saea_steps),
            "gp_restarts": int(run_args.gp_restarts),
            "reference_point": None if ref_point is None else np.asarray(ref_point, dtype=float).tolist(),
            "initial_hv": float(hv_history[0]) if hv_history else float("nan"),
            "final_hv": float(hv_history[-1]) if hv_history else float("nan"),
            "fe_history": fe_history,
            "hv_history": hv_history,
            "comparison_family": "solver_vs_baseline",
            "solver_family": "moead_ego",
            "run_variant": "baseline",
            "history": result["logs"],
        }
        wall_clock_sec = time.perf_counter() - run_started_at
        summary["wall_clock_sec"] = float(wall_clock_sec)
        logger(f"wall_clock_sec = {wall_clock_sec:.6f}")

        plot_results(args=run_args, archive_y=result["Y"], fe_history=fe_history, hv_history=hv_history)

        logger(f"Final FE: {result['X'].shape[0]}")
        logger(f"Final #ND: {result['pareto_Y'].shape[0]}")

        if run_args.output_json:
            output_path = Path(run_args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger(f"output_json = {output_path.resolve()}")

        if run_args.save:
            np.savez_compressed(
                run_args.save,
                X=result["X"],
                Y=result["Y"],
                nd_mask=result["nd_mask"],
                pareto_X=result["pareto_X"],
                pareto_Y=result["pareto_Y"],
                logs=np.asarray(result["logs"], dtype=object),
            )
            logger(f"saved_npz = {Path(run_args.save).resolve()}")
        return summary
    finally:
        log_fp.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="zdt1")
    parser.add_argument("--dim", "--n-var", dest="dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-samples", type=int, default=80)
    parser.add_argument("--infill-budget", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--moead-pop", type=int, default=200)
    parser.add_argument("--saea_steps", type=int, default=30)
    parser.add_argument("--gp-restarts", type=int, default=3)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
