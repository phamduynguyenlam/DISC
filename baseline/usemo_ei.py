from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from problem.problem import SUPPORTED_PROBLEMS, get_reference_point, make_problem
from lhs import latin_hypercube_sample as _scipy_latin_hypercube_sample
from reward import hypervolume, pareto_front
from solver.usemo_solver import run_surrogate_usemo

PLOT_WIDTH_4_3 = 8.0
PLOT_HEIGHT_4_3 = 6.0
WIDE_PLOT_WIDTH_4_3 = 12.0
WIDE_PLOT_HEIGHT_4_3 = 9.0


@dataclass
class USEMOStepRecord:
    step: int
    fe: int
    hv: float
    front_size: int
    selected_idx: int
    selected_uncertainty: float
    selected_true_y: list[float]
    selected_pred_y: list[float]
    selected_sigma: list[float]


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
    return ROOT_DIR.joinpath(*parts)


def default_log_dir(args: argparse.Namespace) -> Path:
    return repo_path("testing_logs", str(args.problem).upper())


def default_log_path(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"usemo_{str(args.problem).lower()}_ei_seed{int(args.seed)}_{timestamp}.txt"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone USeMO baseline with uncertainty-based candidate selection."
    )
    parser.add_argument("--problem", type=str, default="DTLZ6", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--saea_steps", type=int, default=30, metavar="SAEA_STEPS")
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    return args


def latin_hypercube_sample(
    *,
    n_samples: int,
    dim: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    seed: int,
) -> np.ndarray:
    return _scipy_latin_hypercube_sample(
        n_samples=int(n_samples), dim=int(dim), lower=lower, upper=upper, seed=int(seed)
    )


def make_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def aggregate_uncertainty(sigma: np.ndarray) -> np.ndarray:
    sigma_arr = np.asarray(sigma, dtype=np.float32)
    return np.prod(np.maximum(sigma_arr, 1e-12), axis=1).astype(np.float32)


def plot_results(*, args: argparse.Namespace, archive_y: np.ndarray, fe_history: list[int], hv_history: list[float]) -> str | None:
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
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

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4, color="#2ca02c")
    ax_hv.set_title("Hypervolume")
    ax_hv.set_xlabel("Function Evaluations")
    ax_hv.set_ylabel("HV")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y_arr[:, 0], archive_y_arr[:, 1], s=18, alpha=0.35, label="Archive", color="#98df8a")
        ax_pf.scatter(front[:, 0], front[:, 1], s=28, label="Final Front", color="#2ca02c")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
    else:
        ax_pf.scatter(archive_y_arr[:, 0], archive_y_arr[:, 1], archive_y_arr[:, 2], s=18, alpha=0.35, label="Archive", color="#98df8a")
        ax_pf.scatter(front[:, 0], front[:, 1], front[:, 2], s=28, label="Final Front", color="#2ca02c")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    ax_pf.set_title("Final Pareto Front")
    ax_pf.legend(loc="best")

    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return None


def run_once(args: argparse.Namespace) -> dict:
    run_started_at = time.perf_counter()
    problem = make_problem(args.problem, dim=int(args.dim))
    log_path = Path(args.log_path) if args.log_path else default_log_path(args)
    logger, log_fp = make_logger(log_path)

    try:
        logger(
            f"test | baseline = usemo_ei | problem = {args.problem} | "
            f"dim = {int(args.dim)} | seed = {int(args.seed)}"
        )
        archive_x = latin_hypercube_sample(
            n_samples=int(args.init_fe),
            dim=int(problem.dim),
            lower=problem.lower,
            upper=problem.upper,
            seed=int(args.seed),
        )
        archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
        n_obj = int(archive_y.shape[1])
        ref_point = get_reference_point(args.problem, n_obj=n_obj)
        nsga_problem = make_problem_adapter(problem, n_obj)

        hv_history: list[float] = [hypervolume(archive_y, ref_point)]
        fe_history: list[int] = [int(args.init_fe)]
        records: list[USEMOStepRecord] = []

        n_evo_steps = int(args.max_fe) - int(args.init_fe)
        for step in range(n_evo_steps):
            pareto_x, pareto_pred, pareto_sigma = run_surrogate_usemo(
                problem=nsga_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                pop_size=int(args.offspring_size),
                saea_steps=int(args.saea_steps),
                seed=int(args.seed) + int(step),
            )
            uncertainty = aggregate_uncertainty(pareto_sigma)
            selected_idx = int(np.argmax(uncertainty))
            selected_x = pareto_x[selected_idx : selected_idx + 1]
            selected_true_y = np.asarray(problem.evaluate(selected_x), dtype=np.float32).reshape(1, -1)

            archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
            archive_y = np.vstack([archive_y, selected_true_y]).astype(np.float32)

            hv = hypervolume(archive_y, ref_point)
            hv_history.append(float(hv))
            front_size = int(pareto_front(archive_y).shape[0])
            fe = int(args.init_fe) + step + 1
            fe_history.append(int(fe))
            record = USEMOStepRecord(
                step=step + 1,
                fe=fe,
                hv=float(hv),
                front_size=front_size,
                selected_idx=selected_idx,
                selected_uncertainty=float(uncertainty[selected_idx]),
                selected_true_y=selected_true_y.reshape(-1).astype(np.float32).tolist(),
                selected_pred_y=pareto_pred[selected_idx].reshape(-1).astype(np.float32).tolist(),
                selected_sigma=pareto_sigma[selected_idx].reshape(-1).astype(np.float32).tolist(),
            )
            records.append(record)

        summary = {
            "method": "usemo_ei",
            "problem": str(args.problem),
            "dim": int(args.dim),
            "seed": int(args.seed),
            "init_fe": int(args.init_fe),
            "max_fe": int(args.max_fe),
            "offspring_size": int(args.offspring_size),
            "surrogate_model": "gp",
            "saea_steps": int(args.saea_steps),
            "acquisition": "ei",
            "reference_point": ref_point.astype(np.float32).tolist(),
            "initial_hv": float(hv_history[0]),
            "final_hv": float(hv_history[-1]),
            "comparison_family": "solver_vs_baseline",
            "solver_family": "usemo_ei",
            "run_variant": "baseline",
            "fe_history": fe_history,
            "hv_history": [float(v) for v in hv_history],
            "history": [asdict(record) for record in records],
        }
        wall_clock_sec = time.perf_counter() - run_started_at
        summary["wall_clock_sec"] = float(wall_clock_sec)
        logger(
            f"result | FE = {fe_history[-1]} | front = {int(pareto_front(archive_y).shape[0])} | "
            f"HV = {hv_history[-1]:.12f} | wall_sec = {wall_clock_sec:.2f}"
        )
        plot_results(args=args, archive_y=archive_y, fe_history=fe_history, hv_history=hv_history)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        log_fp.close()


def main() -> None:
    args = parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
