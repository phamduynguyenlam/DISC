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

from lhs import latin_hypercube_sample
from problem.problem import SUPPORTED_PROBLEMS, get_reference_point, make_problem
from reward import hypervolume, pareto_front

PLOT_WIDTH_4_3 = 8.0
PLOT_HEIGHT_4_3 = 6.0
WIDE_PLOT_WIDTH_4_3 = 12.0
WIDE_PLOT_HEIGHT_4_3 = 9.0


def _import_pymoo():
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.callback import Callback
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
    except Exception as exc:
        raise ImportError(
            "baseline/nsga2.py requires pymoo. Install pymoo before running this baseline."
        ) from exc

    return {
        "NSGA2": NSGA2,
        "Callback": Callback,
        "Problem": Problem,
        "minimize": minimize,
    }


@dataclass
class NSGA2StepRecord:
    step: int
    fe: int
    hv: float
    front_size: int
    population_size: int


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
    stem = f"nsga2_{str(args.problem).lower()}_seed{int(args.seed)}_{timestamp}.txt"
    return default_log_dir(args) / stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone NSGA-II baseline with pymoo.")
    parser.add_argument("--problem", type=str, default="DTLZ6", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--pop_size", type=int, default=80)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    args = parser.parse_args()
    if int(args.init_fe) != int(args.pop_size):
        raise ValueError(
            f"NSGA-II baseline expects init_fe == pop_size because the initial population consumes the first evaluations, got init_fe={args.init_fe}, pop_size={args.pop_size}."
        )
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    return args


def _as_bounds(problem) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(problem.lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(problem.upper, dtype=np.float64).reshape(-1)
    if lower.size == 1:
        lower = np.repeat(lower, int(problem.dim))
    if upper.size == 1:
        upper = np.repeat(upper, int(problem.dim))
    return lower.astype(np.float64), upper.astype(np.float64)


def plot_results(*, args: argparse.Namespace, population_y: np.ndarray, fe_history: list[int], hv_history: list[float]) -> None:
    pop_y = np.asarray(population_y, dtype=np.float32)
    if pop_y.ndim != 2 or pop_y.shape[0] <= 0:
        return
    n_obj = int(pop_y.shape[1])
    if n_obj not in (2, 3):
        return

    front = pareto_front(pop_y)
    fig = plt.figure(figsize=(WIDE_PLOT_WIDTH_4_3, WIDE_PLOT_HEIGHT_4_3))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4, color="#7f7f7f")
    ax_hv.set_title("Hypervolume")
    ax_hv.set_xlabel("Function Evaluations")
    ax_hv.set_ylabel("HV")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(pop_y[:, 0], pop_y[:, 1], s=18, alpha=0.35, label="Population", color="#c7c7c7")
        ax_pf.scatter(front[:, 0], front[:, 1], s=28, label="Final Front", color="#7f7f7f")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
    else:
        ax_pf.scatter(pop_y[:, 0], pop_y[:, 1], pop_y[:, 2], s=18, alpha=0.35, label="Population", color="#c7c7c7")
        ax_pf.scatter(front[:, 0], front[:, 1], front[:, 2], s=28, label="Final Front", color="#7f7f7f")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    ax_pf.set_title("Final Pareto Front")
    ax_pf.legend(loc="best")

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def run_once(args: argparse.Namespace) -> dict:
    pymoo_mod = _import_pymoo()
    Problem = pymoo_mod["Problem"]
    Callback = pymoo_mod["Callback"]
    NSGA2 = pymoo_mod["NSGA2"]
    minimize = pymoo_mod["minimize"]

    run_started_at = time.perf_counter()
    problem = make_problem(args.problem, dim=int(args.dim))
    lower, upper = _as_bounds(problem)
    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(problem.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(args.seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    sample_y = np.asarray(problem.evaluate(np.asarray(lower, dtype=np.float32).reshape(1, -1)), dtype=np.float32)
    n_obj = int(archive_y.shape[1] if archive_y.ndim == 2 and archive_y.shape[0] > 0 else sample_y.shape[1])
    ref_point = np.asarray(get_reference_point(args.problem, n_obj=n_obj), dtype=np.float32)
    log_path = Path(args.log_path) if args.log_path else default_log_path(args)
    logger, log_fp = make_logger(log_path)
    prefix = "[NSGA2] "

    class WrappedProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=int(problem.dim),
                n_obj=int(n_obj),
                xl=lower,
                xu=upper,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = np.asarray(problem.evaluate(x), dtype=np.float64)

    class HVCallback(Callback):
        def __init__(self):
            super().__init__()
            init_hv = float(hypervolume(archive_y, ref_point))
            init_front_size = int(pareto_front(archive_y).shape[0])
            self.records: list[NSGA2StepRecord] = [
                NSGA2StepRecord(
                    step=0,
                    fe=int(args.init_fe),
                    hv=init_hv,
                    front_size=init_front_size,
                    population_size=int(archive_y.shape[0]),
                )
            ]
            self.fe_history: list[int] = [int(args.init_fe)]
            self.hv_history: list[float] = [init_hv]
            self.population_x: np.ndarray | None = np.asarray(archive_x, dtype=np.float32)
            self.population_y: np.ndarray | None = np.asarray(archive_y, dtype=np.float32)
            self._seen_fe: set[int] = {int(args.init_fe)}

        def notify(self, algorithm):
            raw_fe = int(getattr(algorithm.evaluator, "n_eval", 0))
            fe = max(int(args.init_fe), raw_fe)
            if fe in self._seen_fe:
                return
            pop_x = np.asarray(algorithm.pop.get("X"), dtype=np.float32)
            pop_y = np.asarray(algorithm.pop.get("F"), dtype=np.float32)
            if pop_y.ndim != 2 or pop_y.shape[0] <= 0:
                return
            self._seen_fe.add(fe)
            hv = float(hypervolume(pop_y, ref_point))
            front_size = int(pareto_front(pop_y).shape[0])
            record = NSGA2StepRecord(
                step=len(self.records),
                fe=fe,
                hv=hv,
                front_size=front_size,
                population_size=int(pop_y.shape[0]),
            )
            self.records.append(record)
            self.fe_history.append(int(fe))
            self.hv_history.append(float(hv))
            self.population_x = pop_x
            self.population_y = pop_y
            logger(
                f"{prefix}iter {record.step} | FE = {record.fe} | front = {record.front_size} | "
                f"HV = {record.hv:.12f} | pop = {record.population_size}"
            )

    try:
        logger(f"test_log_path = {log_path.resolve()}")
        logger(f"dim = {int(args.dim)}")
        logger("framework_label = NSGA-II")
        logger("infill_label = NSGA-II")
        logger("comparison_family = solver_vs_baseline")
        logger("solver_family = nsga2")
        logger("run_variant = baseline")
        logger("candidate_solver = nsga2")
        logger("surrogate_model = -")
        logger(f"problem = {args.problem} | dim = {int(args.dim)} | init_fe = {int(args.init_fe)} | max_fe = {int(args.max_fe)}")
        logger(f"pop_size = {int(args.pop_size)}")
        logger("n_offsprings = 1")
        logger(f"reference_point = {ref_point.astype(float).tolist()} (from problem/problem.py)")
        logger(f"{prefix}iter 0 | FE = {int(args.init_fe)} | front = {int(pareto_front(archive_y).shape[0])} | HV = {float(hypervolume(archive_y, ref_point)):.12f} | pop = {int(archive_y.shape[0])}")

        callback = HVCallback()
        algorithm = NSGA2(
            pop_size=int(args.pop_size),
            n_offsprings=1,
            sampling=np.asarray(archive_x, dtype=np.float64),
        )
        result = minimize(
            WrappedProblem(),
            algorithm,
            termination=("n_eval", int(args.max_fe)),
            seed=int(args.seed),
            verbose=False,
            callback=callback,
            save_history=False,
        )

        final_x = np.asarray(result.pop.get("X"), dtype=np.float32)
        final_y = np.asarray(result.pop.get("F"), dtype=np.float32)
        if callback.population_y is None:
            callback.population_x = final_x
            callback.population_y = final_y
            hv = float(hypervolume(final_y, ref_point))
            callback.fe_history.append(int(args.max_fe))
            callback.hv_history.append(hv)
            callback.records.append(
                NSGA2StepRecord(
                    step=0,
                    fe=int(args.max_fe),
                    hv=hv,
                    front_size=int(pareto_front(final_y).shape[0]),
                    population_size=int(final_y.shape[0]),
                )
            )

        summary = {
            "method": "nsga2",
            "backend": "pymoo",
            "problem": str(args.problem),
            "dim": int(args.dim),
            "seed": int(args.seed),
            "init_fe": int(args.init_fe),
            "max_fe": int(args.max_fe),
            "pop_size": int(args.pop_size),
            "reference_point": ref_point.astype(float).tolist(),
            "initial_hv": float(callback.hv_history[0]),
            "final_hv": float(callback.hv_history[-1]),
            "comparison_family": "solver_vs_baseline",
            "solver_family": "nsga2",
            "run_variant": "baseline",
            "fe_history": [int(v) for v in callback.fe_history],
            "hv_history": [float(v) for v in callback.hv_history],
            "history": [asdict(record) for record in callback.records],
            "final_population_x": np.asarray(callback.population_x, dtype=np.float32).tolist(),
            "final_population_y": np.asarray(callback.population_y, dtype=np.float32).tolist(),
        }
        wall_clock_sec = time.perf_counter() - run_started_at
        summary["wall_clock_sec"] = float(wall_clock_sec)
        logger(f"wall_clock_sec = {wall_clock_sec:.6f}")

        plot_results(
            args=args,
            population_y=np.asarray(callback.population_y, dtype=np.float32),
            fe_history=summary["fe_history"],
            hv_history=summary["hv_history"],
        )
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            logger(f"output_json = {output_path.resolve()}")
        return summary
    finally:
        log_fp.close()


def main() -> None:
    args = parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
