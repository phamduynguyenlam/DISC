from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lhs import latin_hypercube_sample
from problem.problem import SUPPORTED_PROBLEMS, get_reference_point, make_problem
from reward import hypervolume, pareto_front
from baseline.cdm_psl_impl.cdm_psl_solver import query_cdm_psl
from surrogate.surrogate_model import fit_gp_surrogates

PLOT_WIDTH_4_3 = 8.0
PLOT_HEIGHT_4_3 = 6.0
WIDE_PLOT_WIDTH_4_3 = 12.0
WIDE_PLOT_HEIGHT_4_3 = 9.0


@dataclass
class CDMPSLStepRecord:
    step: int
    fe: int
    hv: float
    front_size: int
    selected_idx: int
    used_diffusion: bool
    fallback_used: bool
    selected_true_y: list[float]
    selected_pred_y: list[float]
    selected_std: list[float]


class CDMPSLSwitchController:
    def __init__(
        self,
        *,
        initial_use_cdm: bool = True,
        window: int = 3,
        threshold: float = 0.05,
    ) -> None:
        self.use_cdm = bool(initial_use_cdm)
        self.window = max(2, int(window))
        self.threshold = float(threshold)
        self.round_hv_history: list[float] = []
        self.rounds_since_switch = 0

    def update(self, hv_history: list[float]) -> bool:
        if len(hv_history) <= 0:
            return False
        self.round_hv_history.append(float(hv_history[-1]))
        if len(self.round_hv_history) > self.window:
            self.round_hv_history.pop(0)
        if len(self.round_hv_history) < self.window:
            return False

        avg_hv = float(np.mean(np.asarray(self.round_hv_history[:-1], dtype=np.float64)))
        current_hv = float(self.round_hv_history[-1])
        hv_change = abs((current_hv - avg_hv) / max(abs(avg_hv), 1e-12))

        if self.rounds_since_switch >= self.window - 1 and hv_change < self.threshold:
            self.use_cdm = not self.use_cdm
            self.rounds_since_switch = 0
            return True
        self.rounds_since_switch += 1
        return False


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
    stem = f"cdm_psl_{str(args.problem).lower()}_seed{int(args.seed)}_{timestamp}.txt"
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
    parser = argparse.ArgumentParser(description="Run standalone CDM-PSL baseline.")
    parser.add_argument("--problem", type=str, default="DTLZ6", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=5)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--coef_lcb", type=float, default=0.1)
    parser.add_argument("--sbx_rounds", type=int, default=100)
    parser.add_argument("--diffusion_batch_size", type=int, default=1024)
    parser.add_argument("--diffusion_num_epoch", type=int, default=4000)
    parser.add_argument("--diffusion_guided_samples", type=int, default=10)
    parser.add_argument("--diffusion_random_samples", type=int, default=100)
    parser.add_argument("--switch_window", type=int, default=3)
    parser.add_argument("--switch_threshold", type=float, default=0.05)
    parser.add_argument("--start_with_ga", action="store_true")
    parser.add_argument(
        "--disable_diffusion_model",
        "--disable-diffusion-model",
        action="store_true",
        help="Disable CDM diffusion generation and run GA/SBX-only, matching the w/o DM ablation style.",
    )
    parser.add_argument("--exception_fallback", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    if int(args.batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}.")
    if int(args.diffusion_batch_size) <= 0:
        raise ValueError(f"diffusion_batch_size must be positive, got {args.diffusion_batch_size}.")
    return args


def make_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def _bounds_from_problem(problem) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(getattr(problem, "xl", np.zeros(int(problem.n_var))), dtype=np.float32).reshape(-1)
    upper = np.asarray(getattr(problem, "xu", np.ones(int(problem.n_var))), dtype=np.float32).reshape(-1)
    if lower.size == 1:
        lower = np.repeat(lower, int(problem.n_var))
    if upper.size == 1:
        upper = np.repeat(upper, int(problem.n_var))
    return lower.astype(np.float32), upper.astype(np.float32)


def _slice_candidates(result: dict, pop_size: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(result["offspring_x"], dtype=np.float32)
    mean = np.asarray(result["offspring_pred_mean"], dtype=np.float32)
    std = np.asarray(result["offspring_pred_std"], dtype=np.float32)
    selected_indices = np.asarray(result.get("selected_indices", []), dtype=np.int64).reshape(-1)
    if pop_size is not None and int(x.shape[0]) > int(pop_size):
        selected_raw = selected_indices.copy()
        x = x[: int(pop_size)].copy()
        mean = mean[: int(pop_size)].copy()
        std = std[: int(pop_size)].copy()
        if selected_raw.size > 0:
            raw_x = np.asarray(result["offspring_x"], dtype=np.float32)
            raw_mean = np.asarray(result["offspring_pred_mean"], dtype=np.float32)
            raw_std = np.asarray(result["offspring_pred_std"], dtype=np.float32)
            remapped = []
            overflow_slots = list(range(int(pop_size) - 1, -1, -1))
            for selected_idx in selected_raw:
                selected_idx = int(selected_idx)
                if selected_idx < int(pop_size):
                    remapped.append(selected_idx)
                    continue
                if not overflow_slots:
                    break
                dst_idx = overflow_slots.pop(0)
                x[dst_idx] = raw_x[selected_idx]
                mean[dst_idx] = raw_mean[selected_idx]
                std[dst_idx] = raw_std[selected_idx]
                remapped.append(dst_idx)
            selected_indices = np.asarray(remapped, dtype=np.int64)
    return x, mean, std, selected_indices


def propose_cdm_psl_candidates(
    *,
    problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    surrogate=None,
    pop_size: int | None = None,
    seed: int = 0,
    use_diffusion: bool = True,
    fallback_to_sbx: bool = True,
    coef_lcb: float = 0.1,
    sbx_rounds: int = 100,
    diffusion_batch_size: int = 1024,
    diffusion_num_epoch: int = 4000,
    diffusion_guided_samples: int = 10,
    diffusion_random_samples: int = 100,
    n_select: int = 1,
    logger: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    archive_x_arr = np.asarray(archive_x, dtype=np.float32)
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    lower, upper = _bounds_from_problem(problem)
    if surrogate is None:
        surrogate = fit_gp_surrogates(
            archive_x=archive_x_arr,
            archive_y=archive_y_arr,
            seed=int(seed),
        )

    common_kwargs = dict(
        archive_x=archive_x_arr,
        archive_y=archive_y_arr,
        xl=lower,
        xu=upper,
        surrogate=surrogate,
        n_select=int(n_select),
        coef_lcb=float(coef_lcb),
        sbx_rounds=int(sbx_rounds),
        diffusion_batch_size=int(diffusion_batch_size),
        diffusion_num_epoch=int(diffusion_num_epoch),
        diffusion_guided_samples=int(diffusion_guided_samples),
        diffusion_random_samples=int(diffusion_random_samples),
        seed=int(seed),
    )

    fallback_used = False
    try:
        result = query_cdm_psl(use_diffusion=bool(use_diffusion), **common_kwargs)
    except Exception as exc:
        if not fallback_to_sbx or not use_diffusion:
            raise
        fallback_used = True
        if logger is not None:
            logger(f"[cdm_psl] diffusion failed; fallback = sbx_pm | reason = {type(exc).__name__}: {exc}")
        result = query_cdm_psl(use_diffusion=False, **common_kwargs)

    x, mean, std, selected_indices = _slice_candidates(result, pop_size)
    info = {
        "selected_indices": selected_indices,
        "used_diffusion": bool(result.get("used_diffusion", bool(use_diffusion))) and not fallback_used,
        "fallback_used": bool(fallback_used),
        "surrogate": surrogate,
    }
    return x, mean, std, info


def _selected_index_from_info(info: dict, n_candidates: int) -> int:
    selected_indices = np.asarray(info.get("selected_indices", []), dtype=np.int64).reshape(-1)
    if selected_indices.size > 0:
        return int(np.clip(int(selected_indices[0]), 0, int(n_candidates) - 1))
    return 0


def _selected_indices_from_info(info: dict, n_candidates: int, batch_size: int) -> np.ndarray:
    selected_indices = np.asarray(info.get("selected_indices", []), dtype=np.int64).reshape(-1)
    if selected_indices.size <= 0:
        selected_indices = np.arange(int(n_candidates), dtype=np.int64)
    selected_indices = np.clip(selected_indices, 0, int(n_candidates) - 1)
    unique_indices = []
    seen = set()
    for idx in selected_indices:
        idx_int = int(idx)
        if idx_int in seen:
            continue
        unique_indices.append(idx_int)
        seen.add(idx_int)
        if len(unique_indices) >= int(batch_size):
            break
    if len(unique_indices) < int(batch_size):
        for idx_int in range(int(n_candidates)):
            if idx_int in seen:
                continue
            unique_indices.append(idx_int)
            if len(unique_indices) >= int(batch_size):
                break
    return np.asarray(unique_indices[: int(batch_size)], dtype=np.int64)


def run_cdm_psl_baseline(args: argparse.Namespace, logger=print) -> dict:
    problem = make_problem(args.problem, dim=int(args.dim))
    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(problem.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(args.seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    n_obj = int(archive_y.shape[1])
    ref_point = np.asarray(get_reference_point(args.problem, n_obj=n_obj), dtype=np.float32)
    solver_problem = make_problem_adapter(problem, n_obj)

    hv_history = [float(hypervolume(archive_y, ref_point))]
    records: list[CDMPSLStepRecord] = []
    diffusion_disabled = bool(getattr(args, "disable_diffusion_model", False))
    switch = None if diffusion_disabled else CDMPSLSwitchController(
        initial_use_cdm=not bool(getattr(args, "start_with_ga", False)),
        window=int(getattr(args, "switch_window", 3)),
        threshold=float(getattr(args, "switch_threshold", 0.05)),
    )
    n_evo_steps = int(args.max_fe) - int(args.init_fe)
    step = 0
    while step < n_evo_steps:
        current_batch = min(int(args.batch_size), n_evo_steps - step)
        use_cdm = False if diffusion_disabled else bool(switch.use_cdm)
        offspring_x, offspring_mean, offspring_std, info = propose_cdm_psl_candidates(
            problem=solver_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            pop_size=int(args.offspring_size),
            seed=int(args.seed) + int(step),
            use_diffusion=use_cdm,
            fallback_to_sbx=bool(getattr(args, "exception_fallback", False)),
            coef_lcb=float(args.coef_lcb),
            sbx_rounds=int(args.sbx_rounds),
            diffusion_batch_size=int(args.diffusion_batch_size),
            diffusion_num_epoch=int(args.diffusion_num_epoch),
            diffusion_guided_samples=int(args.diffusion_guided_samples),
            diffusion_random_samples=int(args.diffusion_random_samples),
            n_select=int(current_batch),
            logger=logger,
        )
        selected_indices = _selected_indices_from_info(info, int(offspring_x.shape[0]), int(current_batch))
        batch_completed = False
        for selected_idx in selected_indices:
            selected_idx = int(selected_idx)
            selected_x = offspring_x[selected_idx : selected_idx + 1]
            selected_true_y = np.asarray(problem.evaluate(selected_x), dtype=np.float32)

            archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
            archive_y = np.vstack([archive_y, selected_true_y]).astype(np.float32)
            step += 1
            hv = float(hypervolume(archive_y, ref_point))
            hv_history.append(hv)
            front_size = int(pareto_front(archive_y).shape[0])
            record = CDMPSLStepRecord(
                step=int(step),
                fe=int(args.init_fe) + int(step),
                hv=hv,
                front_size=front_size,
                selected_idx=int(selected_idx),
                used_diffusion=bool(info["used_diffusion"]),
                fallback_used=bool(info["fallback_used"]),
                selected_true_y=selected_true_y.reshape(-1).astype(float).tolist(),
                selected_pred_y=offspring_mean[selected_idx].reshape(-1).astype(float).tolist(),
                selected_std=offspring_std[selected_idx].reshape(-1).astype(float).tolist(),
            )
            records.append(record)
            batch_completed = True
            if step >= n_evo_steps:
                break
        if switch is not None and batch_completed:
            switch.update(hv_history)

    return {
        "method": "cdm_psl_wo_dm" if diffusion_disabled else "cdm_psl",
        "problem": str(args.problem),
        "dim": int(args.dim),
        "seed": int(args.seed),
        "init_fe": int(args.init_fe),
        "max_fe": int(args.max_fe),
        "batch_size": int(args.batch_size),
        "offspring_size": int(args.offspring_size),
        "diffusion_batch_size": int(args.diffusion_batch_size),
        "diffusion_num_epoch": int(args.diffusion_num_epoch),
        "diffusion_guided_samples": int(args.diffusion_guided_samples),
        "diffusion_random_samples": int(args.diffusion_random_samples),
        "disable_diffusion_model": bool(diffusion_disabled),
        "surrogate_model": "gp",
        "reference_point": ref_point.astype(float).tolist(),
        "switch_window": int(getattr(args, "switch_window", 3)),
        "switch_threshold": float(getattr(args, "switch_threshold", 0.05)),
        "initial_hv": float(hv_history[0]),
        "final_hv": float(hv_history[-1]),
        "fe_history": [int(args.init_fe)] + [int(record.fe) for record in records],
        "hv_history": [float(v) for v in hv_history],
        "history": [asdict(record) for record in records],
        "final_archive_y": archive_y.astype(float).tolist(),
    }


def plot_results(*, args: argparse.Namespace, summary: dict) -> str | None:
    archive_y = np.asarray(summary.get("final_archive_y", []), dtype=np.float32)
    if archive_y.ndim != 2 or archive_y.shape[0] <= 0:
        return None
    n_obj = int(archive_y.shape[1])
    if n_obj not in (2, 3):
        return None

    front = pareto_front(archive_y)
    fe_history = [int(v) for v in summary.get("fe_history", [])]
    hv_history = [float(v) for v in summary.get("hv_history", [])]

    fig = plt.figure(figsize=(WIDE_PLOT_WIDTH_4_3, WIDE_PLOT_HEIGHT_4_3))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4, color="#1f77b4")
    ax_hv.set_title("Hypervolume")
    ax_hv.set_xlabel("Function Evaluations")
    ax_hv.set_ylabel("HV")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], s=18, alpha=0.35, label="Archive", color="#9ecae1")
        ax_pf.scatter(front[:, 0], front[:, 1], s=28, label="Final Front", color="#1f77b4")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
    else:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], archive_y[:, 2], s=18, alpha=0.35, label="Archive", color="#9ecae1")
        ax_pf.scatter(front[:, 0], front[:, 1], front[:, 2], s=28, label="Final Front", color="#1f77b4")
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
    log_path = Path(args.log_path) if args.log_path else default_log_path(args)
    logger, log_fp = make_logger(log_path)
    try:
        framework_label = "CDM-PSL w/o DM" if bool(getattr(args, "disable_diffusion_model", False)) else "CDM-PSL"
        comparison_family = "solver_vs_baseline" if framework_label == "CDM-PSL" else None
        solver_family = "cdm_psl" if framework_label == "CDM-PSL" else None
        run_variant = "baseline" if framework_label == "CDM-PSL" else None
        logger(
            f"test | baseline = {str(framework_label).lower().replace(' ', '_')} | "
            f"problem = {args.problem} | dim = {int(args.dim)} | seed = {int(args.seed)}"
        )
        summary = run_cdm_psl_baseline(args, logger=logger)
        wall_clock_sec = time.perf_counter() - run_started_at
        summary["wall_clock_sec"] = float(wall_clock_sec)
        logger(
            f"result | FE = {summary['fe_history'][-1]} | "
            f"front = {int(pareto_front(np.asarray(summary['final_archive_y'])).shape[0])} | "
            f"HV = {summary['final_hv']:.12f} | wall_sec = {wall_clock_sec:.2f}"
        )
        plot_results(args=args, summary=summary)
        summary_for_json = dict(summary)
        summary_for_json["comparison_family"] = comparison_family
        summary_for_json["solver_family"] = solver_family
        summary_for_json["run_variant"] = run_variant
        summary_for_json.pop("final_archive_y", None)
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary_for_json, indent=2), encoding="utf-8")
        return summary_for_json
    finally:
        log_fp.close()


def main() -> None:
    args = parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
