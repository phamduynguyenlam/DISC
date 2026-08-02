from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from problem.problem import SUPPORTED_PROBLEMS, get_reference_point, make_problem
from lhs import latin_hypercube_sample as _scipy_latin_hypercube_sample
from reward import hypervolume, pareto_front

PLOT_WIDTH_4_3 = 8.0
PLOT_HEIGHT_4_3 = 6.0
WIDE_PLOT_WIDTH_4_3 = 12.0
WIDE_PLOT_HEIGHT_4_3 = 9.0


@dataclass
class QNEHVIStepRecord:
    batch_step: int
    batch_member: int
    fe: int
    hv: float
    front_size: int
    selected_x: list[float]
    selected_true_y: list[float]
    selected_pred_y: list[float]
    selected_std: list[float]


def _import_botorch():
    try:
        from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import ModelListGP, SingleTaskGP
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.constraints import Interval
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    except Exception as exc:
        raise ImportError(
            "baseline/qnehvi.py requires botorch and gpytorch. "
            "Install them before running this baseline."
        ) from exc

    return {
        "ModelListGP": ModelListGP,
        "SingleTaskGP": SingleTaskGP,
        "SumMarginalLogLikelihood": SumMarginalLogLikelihood,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "qNoisyExpectedHypervolumeImprovement": qNoisyExpectedHypervolumeImprovement,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "optimize_acqf": optimize_acqf,
        "Interval": Interval,
        "MaternKernel": MaternKernel,
        "ScaleKernel": ScaleKernel,
    }


def _suppress_qnehvi_botorch_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*qNoisyExpectedHypervolumeImprovement has known numerical issues.*",
        category=Warning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Optimization failed in `gen_candidates_scipy`.*",
        category=RuntimeWarning,
    )


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
    kernel_tag = f"_k{str(getattr(args, 'kernel', 'default')).lower()}"
    stem = (
        f"qnehvi_{str(args.problem).lower()}_q{int(args.q)}_mc{int(args.n_mc)}_"
        f"seed{int(args.seed)}{kernel_tag}_{timestamp}.txt"
    )
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
        description="Run end-to-end qNEHVI baseline with BoTorch, evaluating all q candidates sequentially each batch."
    )
    parser.add_argument("--problem", type=str, default="DTLZ6", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="double", choices=["float", "double"])
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--kernel", type=str, default="default", choices=["default", "common"])
    parser.add_argument("--n_mc", type=int, default=128)
    parser.add_argument("--num_restarts", type=int, default=10)
    parser.add_argument("--raw_samples", type=int, default=256)
    parser.add_argument("--acq_maxiter", type=int, default=200)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()
    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    q_value = int(args.q)
    n_evals = int(args.max_fe) - int(args.init_fe)
    if q_value <= 0:
        raise ValueError(f"q must be positive, got {q_value}.")
    if n_evals % q_value != 0:
        raise ValueError(
            f"max_fe - init_fe must be divisible by q for this baseline, got {n_evals} and q={q_value}."
        )
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


def _as_torch_dtype(dtype_name: str):
    return torch.double if str(dtype_name).lower() == "double" else torch.float32


def _make_bounds(problem) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(problem.lower, dtype=np.float32).reshape(-1)
    upper = np.asarray(problem.upper, dtype=np.float32).reshape(-1)
    if lower.size == 1:
        lower = np.repeat(lower, int(problem.dim))
    if upper.size == 1:
        upper = np.repeat(upper, int(problem.dim))
    return lower.astype(np.float32), upper.astype(np.float32)


def _normalize_x(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    lower_arr = np.asarray(lower, dtype=np.float32).reshape(1, -1)
    upper_arr = np.asarray(upper, dtype=np.float32).reshape(1, -1)
    span = np.maximum(upper_arr - lower_arr, 1e-12)
    return ((x_arr - lower_arr) / span).astype(np.float32)


def _unnormalize_x(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    lower_arr = np.asarray(lower, dtype=np.float32).reshape(1, -1)
    upper_arr = np.asarray(upper, dtype=np.float32).reshape(1, -1)
    return (lower_arr + x_arr * (upper_arr - lower_arr)).astype(np.float32)


def fit_botorch_surrogate(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    device: torch.device,
    torch_dtype,
    kernel: str = "default",
):
    botorch_mod = _import_botorch()
    train_x_t = torch.as_tensor(np.asarray(train_x, dtype=np.float32), device=device, dtype=torch_dtype)
    train_y_t = torch.as_tensor(np.asarray(train_y, dtype=np.float32), device=device, dtype=torch_dtype)

    models = []
    for obj_idx in range(int(train_y_t.shape[1])):
        covar_module = None
        if str(kernel).lower() == "common":
            covar_module = botorch_mod["ScaleKernel"](
                botorch_mod["MaternKernel"](
                    ard_num_dims=int(train_x_t.shape[-1]),
                    nu=2.5,
                    lengthscale_constraint=botorch_mod["Interval"](
                        float(np.sqrt(1e-3)),
                        float(np.sqrt(1e3)),
                    ),
                )
            )
        model = botorch_mod["SingleTaskGP"](
            train_X=train_x_t,
            train_Y=train_y_t[:, obj_idx : obj_idx + 1],
            covar_module=covar_module,
        )
        models.append(model)

    model_list = botorch_mod["ModelListGP"](*models).to(device=device, dtype=torch_dtype)
    mll = botorch_mod["SumMarginalLogLikelihood"](model_list.likelihood, model_list)
    botorch_mod["fit_gpytorch_mll"](mll)
    return model_list


def generate_qnehvi_candidates(
    *,
    model,
    train_x_norm: np.ndarray,
    ref_point: np.ndarray,
    q: int,
    n_mc: int,
    lower: np.ndarray,
    upper: np.ndarray,
    device: torch.device,
    torch_dtype,
    num_restarts: int,
    raw_samples: int,
    acq_maxiter: int,
):
    botorch_mod = _import_botorch()
    sampler = botorch_mod["SobolQMCNormalSampler"](sample_shape=torch.Size([int(n_mc)]))
    x_baseline = torch.as_tensor(np.asarray(train_x_norm, dtype=np.float32), device=device, dtype=torch_dtype)
    with warnings.catch_warnings():
        _suppress_qnehvi_botorch_warnings()
        acqf = botorch_mod["qNoisyExpectedHypervolumeImprovement"](
            model=model,
            ref_point=np.asarray(ref_point, dtype=np.float32).reshape(-1).astype(float).tolist(),
            X_baseline=x_baseline,
            sampler=sampler,
            prune_baseline=False,
        )
    dim = int(train_x_norm.shape[1])
    bounds = torch.stack(
        [
            torch.zeros(dim, device=device, dtype=torch_dtype),
            torch.ones(dim, device=device, dtype=torch_dtype),
        ],
        dim=0,
    )
    with warnings.catch_warnings():
        _suppress_qnehvi_botorch_warnings()
        candidates_norm, acq_value = botorch_mod["optimize_acqf"](
            acq_function=acqf,
            bounds=bounds,
            q=int(q),
            num_restarts=int(num_restarts),
            raw_samples=int(raw_samples),
            options={"batch_limit": 5, "maxiter": int(acq_maxiter)},
            sequential=True,
        )
    candidates_norm_np = np.asarray(candidates_norm.detach().cpu().numpy(), dtype=np.float32).reshape(int(q), dim)
    posterior = model.posterior(candidates_norm.to(device=device, dtype=torch_dtype))
    pred_mean = -np.asarray(posterior.mean.detach().cpu().numpy(), dtype=np.float32).reshape(int(q), -1)
    pred_std = np.asarray(posterior.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy(), dtype=np.float32).reshape(int(q), -1)
    candidates_x = _unnormalize_x(candidates_norm_np, lower, upper)
    acq_scalar = float(acq_value.detach().cpu().reshape(-1)[0].item()) if hasattr(acq_value, "detach") else float(acq_value)
    return candidates_x, pred_mean.astype(np.float32), pred_std.astype(np.float32), acq_scalar


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

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4, color="#d62728")
    ax_hv.set_title("Hypervolume")
    ax_hv.set_xlabel("Function Evaluations")
    ax_hv.set_ylabel("HV")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y_arr[:, 0], archive_y_arr[:, 1], s=18, alpha=0.35, label="Archive", color="#f7b6d2")
        ax_pf.scatter(front[:, 0], front[:, 1], s=28, label="Final Front", color="#d62728")
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
    else:
        ax_pf.scatter(archive_y_arr[:, 0], archive_y_arr[:, 1], archive_y_arr[:, 2], s=18, alpha=0.35, label="Archive", color="#f7b6d2")
        ax_pf.scatter(front[:, 0], front[:, 1], front[:, 2], s=28, label="Final Front", color="#d62728")
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
    torch_dtype = _as_torch_dtype(args.dtype)
    device = torch.device(str(args.device))
    problem = make_problem(args.problem, dim=int(args.dim))
    lower, upper = _make_bounds(problem)
    log_path = Path(args.log_path) if args.log_path else default_log_path(args)
    logger, log_fp = make_logger(log_path)
    prefix = "[QNEHVI] "

    try:
        logger(f"log_path = {log_path.resolve()}")
        logger(f"dim = {int(args.dim)}")
        logger("framework_label = qNEHVI")
        logger("infill_label = qNEHVI")
        logger("comparison_family = solver_vs_baseline")
        logger("solver_family = qnehvi")
        logger("run_variant = baseline")
        logger("backend = botorch-qNEHVI")
        logger(f"kernel = {str(args.kernel).lower()}")
        logger(f"problem = {args.problem} | dim = {int(args.dim)} | init_fe = {int(args.init_fe)} | max_fe = {int(args.max_fe)}")
        logger(f"q = {int(args.q)} | n_mc = {int(args.n_mc)} | num_restarts = {int(args.num_restarts)} | raw_samples = {int(args.raw_samples)}")

        archive_x = latin_hypercube_sample(
            n_samples=int(args.init_fe),
            dim=int(problem.dim),
            lower=lower,
            upper=upper,
            seed=int(args.seed),
        )
        archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
        n_obj = int(archive_y.shape[1])
        ref_point_min = np.asarray(get_reference_point(args.problem, n_obj=n_obj), dtype=np.float32)
        ref_point_max = (-ref_point_min).astype(np.float32)
        hv_history: list[float] = [float(hypervolume(archive_y, ref_point_min))]
        fe_history: list[int] = [int(args.init_fe)]
        records: list[QNEHVIStepRecord] = []
        logger(f"reference_point = {ref_point_min.astype(float).tolist()} (minimization HV)")
        logger(f"{prefix}iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.12f}")

        n_batches = (int(args.max_fe) - int(args.init_fe)) // int(args.q)
        fe = int(args.init_fe)
        for batch_step in range(int(n_batches)):
            train_x_norm = _normalize_x(archive_x, lower, upper)
            train_y_max = (-np.asarray(archive_y, dtype=np.float32)).astype(np.float32)
            model = fit_botorch_surrogate(
                train_x=train_x_norm,
                train_y=train_y_max,
                device=device,
                torch_dtype=torch_dtype,
                kernel=str(args.kernel).lower(),
            )
            batch_x, batch_pred_mean, batch_pred_std, acq_scalar = generate_qnehvi_candidates(
                model=model,
                train_x_norm=train_x_norm,
                ref_point=ref_point_max,
                q=int(args.q),
                n_mc=int(args.n_mc),
                lower=lower,
                upper=upper,
                device=device,
                torch_dtype=torch_dtype,
                num_restarts=int(args.num_restarts),
                raw_samples=int(args.raw_samples),
                acq_maxiter=int(args.acq_maxiter),
            )
            logger(
                f"batch {batch_step + 1:02d} | qnehvi_acq = {acq_scalar:.6f} | "
                f"batch_candidates = {int(batch_x.shape[0])}"
            )

            for member_idx in range(int(batch_x.shape[0])):
                selected_x = batch_x[member_idx : member_idx + 1]
                selected_true_y = np.asarray(problem.evaluate(selected_x), dtype=np.float32)
                archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
                archive_y = np.vstack([archive_y, selected_true_y]).astype(np.float32)
                fe += 1
                hv = float(hypervolume(archive_y, ref_point_min))
                hv_history.append(hv)
                fe_history.append(int(fe))
                front_size = int(pareto_front(archive_y).shape[0])
                record = QNEHVIStepRecord(
                    batch_step=int(batch_step) + 1,
                    batch_member=int(member_idx) + 1,
                    fe=int(fe),
                    hv=hv,
                    front_size=front_size,
                    selected_x=selected_x.reshape(-1).astype(float).tolist(),
                    selected_true_y=selected_true_y.reshape(-1).astype(float).tolist(),
                    selected_pred_y=batch_pred_mean[member_idx].reshape(-1).astype(float).tolist(),
                    selected_std=batch_pred_std[member_idx].reshape(-1).astype(float).tolist(),
                )
                records.append(record)
                logger(
                    f"{prefix}batch {record.batch_step:02d} cand {record.batch_member} | "
                    f"FE = {record.fe} | front = {record.front_size} | HV = {record.hv:.12f}"
                )

                if member_idx < int(batch_x.shape[0]) - 1:
                    train_x_norm = _normalize_x(archive_x, lower, upper)
                    train_y_max = (-np.asarray(archive_y, dtype=np.float32)).astype(np.float32)
                    model = fit_botorch_surrogate(
                        train_x=train_x_norm,
                        train_y=train_y_max,
                        device=device,
                        torch_dtype=torch_dtype,
                        kernel=str(args.kernel).lower(),
                    )

        summary = {
            "method": "qnehvi",
            "backend": "botorch",
            "problem": str(args.problem),
            "dim": int(args.dim),
            "seed": int(args.seed),
            "kernel": str(args.kernel).lower(),
            "init_fe": int(args.init_fe),
            "max_fe": int(args.max_fe),
            "q": int(args.q),
            "n_mc": int(args.n_mc),
            "num_restarts": int(args.num_restarts),
            "raw_samples": int(args.raw_samples),
            "reference_point": ref_point_min.astype(float).tolist(),
            "initial_hv": float(hv_history[0]),
            "final_hv": float(hv_history[-1]),
            "comparison_family": "solver_vs_baseline",
            "solver_family": "qnehvi",
            "run_variant": "baseline",
            "fe_history": fe_history,
            "hv_history": [float(v) for v in hv_history],
            "history": [asdict(record) for record in records],
        }
        wall_clock_sec = time.perf_counter() - run_started_at
        summary["wall_clock_sec"] = float(wall_clock_sec)
        logger(f"wall_clock_sec = {wall_clock_sec:.6f}")
        plot_results(args=args, archive_y=archive_y, fe_history=fe_history, hv_history=hv_history)
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
