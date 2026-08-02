from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import secrets
import sys
import time
import textwrap
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.db_saea import DBSAEAAgent
from agents.boformer import BOFormer, upgrade_legacy_observation_action_weight
from agents.disc import Disc, DiscAF, DiscAF2
from agents.disc_single_dqn import DiscSingleDQN
from baseline.moead_ego import propose_moead_ego_candidates
from infill import (
    EPDIExploitation,
    EPDIExploration,
    ExpectedHypervolumeImprovement,
    NDA,
    NDPBIConvergence,
    NDPBIDiversity,
    RandomSelection,
    SurrogateParetoImprovement,
    TchebycheffProbabilityOfImprovement,
    USeMOUncertainty,
)
from lhs import latin_hypercube_sample
from solver.hybrid_solver import run_surrogate_hybrid
from solver.moead_ego_solver import run_surrogate_moead_ego
from problem.problem import (
    SUPPORTED_PROBLEMS,
    default_problem_dim,
    get_reference_point,
    get_true_pareto_hv,
    make_problem,
)
from reward import (
    hypervolume,
    pareto_front,
    resolve_default_reward_lambda,
    reward_scheme_1,
    reward_scheme_2,
)
from surrogate.surrogate_model import (
    estimate_uncertainty,
    fit_gp_surrogates,
    surrogate_model_name,
)


class _NullLogFile:
    def close(self) -> None:
        return None


BOFORMER_N_LAYERS = 8
BOFORMER_N_HEADS = 4
BOFORMER_WINDOW_SIZE = 31
BOFORMER_DROPOUT = 0.1
BOFORMER_TEMPERATURE = 1000.0


def make_test_logger(log_path: Path | None):
    if log_path is None:
        def _log(message: str) -> None:
            print(str(message))

        return _log, _NullLogFile()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("w", encoding="utf-8")

    def _log(message: str) -> None:
        text = str(message)
        print(text)
        log_fp.write(text + "\n")
        log_fp.flush()

    return _log, log_fp


def default_test_log_dir(args: argparse.Namespace) -> Path:
    agent_pth = str(getattr(args, "agent_pth", "") or "").strip().lower()
    if "_rs2_" in agent_pth:
        return Path("testing_logs") / "rs2" / str(args.problem).upper()
    return Path("testing_logs") / str(args.problem).upper()


def default_test_plot_dir(args: argparse.Namespace, *, kind: str) -> Path:
    return Path(str(kind)) / str(args.problem).upper()


def default_test_log_path(args: argparse.Namespace, *, agent_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    compare_name = resolve_compare_infill_name(args)
    compare_algo_name = resolve_compare_algo_name(args)
    compare_parts = []
    if compare_name is not None:
        compare_parts.append(str(compare_name))
    if compare_algo_name is not None:
        compare_parts.append(f"algo_{compare_algo_name}")
    compare_tag = f"_compare_{'_'.join(compare_parts)}" if compare_parts else ""
    primary_tag = resolve_primary_output_tag(args, agent_name=agent_name)
    stem = (
        f"test_{primary_tag}_{str(args.problem).lower()}_"
        f"{str(args.surrogate_model).lower()}_seed{int(args.seed)}"
        f"{compare_tag}_{timestamp}.txt"
    )
    return default_test_log_dir(args) / stem


def resolve_primary_output_tag(args: argparse.Namespace, *, agent_name: str | None = None) -> str:
    primary_policy = resolve_primary_policy_name(args)
    if primary_policy == "disc":
        solver_name = str(getattr(args, "solver", "")).strip().lower().replace("-", "_")
        solver_family = canonical_solver_family_name(solver_name)
        agent_tag = str(agent_name if agent_name is not None else getattr(args, "agent_name", "disc")).lower()
        if solver_family is not None:
            return f"{agent_tag}_solver_{solver_family}"
        return agent_tag
    return str(primary_policy).lower()


def _normalize_display_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace("/", "_")


def resolve_compare_infill_name(args: argparse.Namespace) -> str | None:
    names = resolve_compare_infill_names(args)
    if len(names) <= 0:
        return None
    return ",".join(names)


def resolve_compare_algo_name(args: argparse.Namespace) -> str | None:
    names = resolve_compare_algo_names(args)
    if len(names) <= 0:
        return None
    return ",".join(names)


def resolve_compare_infill_names(args: argparse.Namespace) -> list[str]:
    raw_value = getattr(args, "compare_infill", None)
    if raw_value is None:
        return []
    raw_text = str(raw_value).strip().lower()
    if raw_text == "":
        return []
    all_names = [
        "db_saea",
        "ehvi",
        "usemo_uncertainty",
        "spi",
        "pi",
        "random",
        "nd_a",
        "nd_pbi_convergence",
        "nd_pbi_diversity",
        "epdi_exploitation",
        "epdi_exploration",
    ]
    allowed = set(all_names + ["all"])
    tokens = [part.strip().lower().replace("-", "_") for part in raw_text.split(",")]
    tokens = [token for token in tokens if token != ""]
    if len(tokens) <= 0:
        return []

    resolved: list[str] = []
    for token in tokens:
        if token not in allowed:
            raise ValueError(
                f"Unsupported compare_infill entry: {token}. "
                "Expected entries from infill.py or all."
            )
        if token == "all":
            for name in all_names:
                if name not in resolved:
                    resolved.append(name)
            continue
        if token not in resolved:
            resolved.append(token)
    return resolved


def resolve_compare_algo_names(args: argparse.Namespace) -> list[str]:
    raw_value = getattr(args, "compare_algo", None)
    if raw_value is None:
        return []
    raw_text = str(raw_value).strip().lower()
    if raw_text == "":
        return []
    allowed = {
        "disc",
        "disc_af",
        "disc_af2",
        "disc_single_dqn",
        "moead_ego",
        "all",
    }
    tokens = [part.strip().lower().replace("-", "_") for part in raw_text.split(",")]
    tokens = [token for token in tokens if token != ""]
    if len(tokens) <= 0:
        return []
    resolved: list[str] = []
    for token in tokens:
        if token not in allowed:
            raise ValueError(
                f"Unsupported compare_algo entry: {token}. "
                "Expected entries from {'disc','disc_af','disc_af2','disc_single_dqn','moead_ego','all'}."
            )
        if token == "all":
            for name in ["moead_ego"]:
                if name not in resolved:
                    resolved.append(name)
            continue
        if token not in resolved:
            resolved.append(token)
    if "disc" in resolved:
        resolved = ["disc"] + [token for token in resolved if token != "disc"]
    return resolved


def resolve_infill_name(args: argparse.Namespace) -> str | None:
    raw_value = getattr(args, "infill", None)
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text == "":
        return None
    return text.replace("-", "_")


def resolve_primary_policy_name(args: argparse.Namespace) -> str:
    infill_name = resolve_infill_name(args)
    agent_name = str(getattr(args, "agent_name", "disc")).strip().lower().replace("-", "_")
    if infill_name is None:
        if agent_name in {"disc_af", "disc_af2", "disc_single_dqn", "boformer"}:
            return agent_name
        return "disc"
    if infill_name == "disc":
        return "disc"
    if infill_name in {"disc_af", "af"}:
        return "disc_af"
    if infill_name in {"disc_af2", "af2"}:
        return "disc_af2"
    if infill_name in {"disc_single_dqn", "single_dqn"}:
        return "disc_single_dqn"
    if infill_name == "boformer":
        return "boformer"
    return str(infill_name)


def primary_policy_display_name(name: str) -> str:
    key = _normalize_display_key(name)
    if key == "disc":
        return "DISC"
    if key == "disc_af":
        return "DISC w/o candidate pool interaction"
    if key == "disc_af2":
        return "DISC-AF2"
    if key == "disc_single_dqn":
        return "DISC Single DQN"
    if key == "boformer":
        return "BOFormer"
    if key == "db_saea":
        return "DB-SAEA"
    return compare_infill_display_name(key)


def is_agent_policy(name: str) -> bool:
    return str(name).strip().lower().replace("-", "_") in {
        "disc",
        "disc_af",
        "disc_af2",
        "disc_single_dqn",
        "boformer",
    }


def is_db_saea_policy(name: str) -> bool:
    return str(name).strip().lower().replace("-", "_") == "db_saea"


def is_learned_infill_policy(name: str) -> bool:
    return bool(is_agent_policy(name) or is_db_saea_policy(name))


def compare_infill_display_name(name: str) -> str:
    display_map = {
        "ehvi": "EHVI",
        "usemo_uncertainty": "USeMO-Uncertainty",
        "spi": "SPI",
        "surrogate_pareto_improvement": "SPI",
        "pi": "PI",
        "tchebycheff_probability_of_improvement": "PI",
        "random": "RANDOM",
        "db_saea": "DB-SAEA",
        "nd_a": "ND-A",
        "nd_pbi_convergence": "ND-PBI-Convergence",
        "nd_pbi_diversity": "ND-PBI-Diversity",
        "epdi_exploitation": "EPDI-Exploitation",
        "epdi_exploration": "EPDI-Exploration",
        "moea_d_ego": "MOEA-D/EGO",
        "moead_ego": "MOEA-D/EGO",
        "boformer": "BOFormer",
        "disc_af2": "DISC-AF2",
        "disc_single_dqn": "DISC Single DQN",
    }
    key = _normalize_display_key(name)
    return display_map.get(key, key.upper())


def compare_algo_display_name(name: str) -> str:
    display_map = {
        "disc": "DISC",
        "boformer": "BOFormer",
        "disc_af": "DISC w/o candidate pool interaction",
        "disc_af2": "DISC-AF2",
        "disc_single_dqn": "DISC Single DQN",
        "db_saea": "DB-SAEA",
        "nsga2": "NSGA-II",
        "moead_ego": "MOEA-D/EGO",
        "moea_d_ego": "MOEA-D/EGO",
    }
    key = _normalize_display_key(name)
    return display_map.get(key, key.upper())


def resolve_framework_label(args: argparse.Namespace) -> str:
    primary_policy = resolve_primary_policy_name(args)
    if is_db_saea_policy(primary_policy):
        return "DB-SAEA"
    return "DISC"


def resolve_infill_label(args: argparse.Namespace) -> str:
    primary_policy = resolve_primary_policy_name(args)
    if is_db_saea_policy(primary_policy):
        return "db-saea"
    if is_learned_infill_policy(primary_policy):
        return primary_policy_display_name(primary_policy)
    return compare_infill_display_name(primary_policy)


def canonical_solver_family_name(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower().replace("-", "_").replace("/", "_")
    if key in {"moead_ego", "moea_d_ego"}:
        return "moead_ego"
    if key == "hybrid":
        return None
    return None


def resolve_solver_baseline_metadata(args: argparse.Namespace) -> tuple[str | None, str | None]:
    framework_label = resolve_framework_label(args)
    solver_name = str(getattr(args, "solver", "")).strip().lower().replace("-", "_")
    solver_family = canonical_solver_family_name(solver_name)
    if framework_label == "DISC" and solver_family is not None:
        return "solver_vs_baseline", "disc_solver"
    return None, None


def build_compare_infill_criterion(name: str, *, ref_point: np.ndarray):
    key = str(name).strip().lower().replace("-", "_")
    if key == "ehvi":
        return ExpectedHypervolumeImprovement(ref_point=ref_point, n_samples=128)
    if key == "usemo_uncertainty":
        return USeMOUncertainty()
    if key in {"spi", "surrogate_pareto_improvement"}:
        return SurrogateParetoImprovement()
    if key in {"pi", "tchebycheff_probability_of_improvement"}:
        return TchebycheffProbabilityOfImprovement()
    if key == "random":
        return RandomSelection()
    if key == "nd_a":
        return NDA()
    if key == "nd_pbi_convergence":
        return NDPBIConvergence()
    if key == "nd_pbi_diversity":
        return NDPBIDiversity()
    if key == "epdi_exploitation":
        return EPDIExploitation()
    if key == "epdi_exploration":
        return EPDIExploration()
    raise ValueError(f"Unsupported compare_infill: {name}")


def _call_infill_select_index(criterion, **kwargs) -> tuple[int, np.ndarray]:
    signature = inspect.signature(criterion.select_index)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        filtered_kwargs = kwargs
    else:
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return criterion.select_index(**filtered_kwargs)


def _make_infill_selection_kwargs(
    *,
    criterion,
    archive_y: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    seed: int,
    offspring_info: dict[str, Any] | None = None,
    progress_ratio: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "archive_y": archive_y,
        "candidate_mean": candidate_mean,
        "candidate_std": candidate_std,
        "seed": int(seed),
    }
    if offspring_info is not None:
        candidate_weight_vectors = offspring_info.get("candidate_weight_vectors")
        if candidate_weight_vectors is not None:
            kwargs["candidate_weight_vectors"] = np.asarray(
                candidate_weight_vectors,
                dtype=np.float32,
            )
    del criterion, progress_ratio
    return kwargs


def resolve_test_reward_scheme(args: argparse.Namespace) -> int:
    agent_pth = getattr(args, "agent_pth", None)
    if not agent_pth:
        return 1
    match = re.search(r"rs(\d+)", Path(str(agent_pth)).name.lower())
    if match is None:
        return 1
    reward_scheme_id = int(match.group(1))
    if reward_scheme_id not in {1, 2}:
        raise ValueError(
            f"Unsupported reward scheme tag rs{reward_scheme_id} in checkpoint filename; "
            "expected rs1 or rs2."
        )
    return reward_scheme_id


def resolve_hybrid_branch_steps(args: argparse.Namespace) -> tuple[int, int]:
    del args
    return 15, 30


def is_random_agent_path(agent_pth: str | None) -> bool:
    return agent_pth is not None and str(agent_pth).strip().lower() in {"random", "random_model"}


def resolve_default_best_reward_checkpoint(args: argparse.Namespace) -> str:
    policy_name = resolve_primary_policy_name(args)
    log_subdir = {
        "disc": "disc",
        "disc_af": "disc_af",
        "disc_af2": "disc_af2",
        "disc_single_dqn": "disc_single_dqn",
        "boformer": "boformer",
        "db_saea": "db-saea",
    }.get(policy_name, policy_name)
    checkpoint_root = Path("training_logs") / str(log_subdir)
    filename_pattern = (
        f"{policy_name}_problem_{str(args.problem).lower()}_*_best_reward.pth"
    )
    candidates = [
        path
        for path in checkpoint_root.rglob(filename_pattern)
        if path.is_file()
    ] if checkpoint_root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"No best-reward checkpoint found for {policy_name} on {args.problem} "
            f"under {checkpoint_root}."
        )
    rs1_candidates = [path for path in candidates if "_rs1_" in path.name.lower()]
    if rs1_candidates:
        candidates = rs1_candidates
    selected = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return str(selected.resolve())


def resolve_inference_source(
    *,
    policy_name: str,
    agent_pth: str | None = None,
    random_model: bool = False,
    compare_algo: str | None = None,
) -> str:
    key = str(policy_name).strip().lower().replace("-", "_")
    if compare_algo is not None:
        algo_name = str(compare_algo).strip().lower().replace("-", "_")
        if bool(random_model):
            return f"{algo_name}:random_model"
        if agent_pth:
            return str(Path(str(agent_pth)).resolve())
        return f"{algo_name}:uninitialized_model"
    if key in {"disc", "disc_af", "disc_af2", "disc_single_dqn", "boformer", "db_saea"}:
        if bool(random_model):
            return f"{key}:random_model"
        if agent_pth:
            return str(Path(str(agent_pth)).resolve())
        return f"{key}:uninitialized_model"
    return f"infill:{key}"


def format_policy_prefix(policy_name: str) -> str:
    return f"[{str(policy_name).strip().lower().replace('-', '_')}] "


def compute_test_reward(
    *,
    reward_scheme_id: int,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    reward_lambda: float,
    true_pareto_hv: float | None = None,
) -> float:
    if int(reward_scheme_id) == 1:
        return float(
            reward_scheme_1(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                reward_lambda=float(reward_lambda),
            )
        )
    if int(reward_scheme_id) == 2:
        if true_pareto_hv is None:
            raise ValueError("reward_scheme_2 requires true_pareto_hv in tester.")
        return float(
            reward_scheme_2(
                previous_front=previous_front,
                selected_objectives=selected_objectives,
                ref_point=ref_point,
                true_pareto_hv=float(true_pareto_hv),
                reward_lambda=float(reward_lambda),
            )
        )
    raise ValueError(f"Unsupported reward_scheme_id for tester: {reward_scheme_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DISC-guided surrogate-assisted optimization with 80 LHS init + 40 evolution steps."
    )
    parser.add_argument("--problem", type=lambda value: str(value).upper(), default="ZDT1", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--seed", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--offspring_size", type=int, default=None)
    parser.add_argument("--logit_scale", type=float, default=5.0)
    parser.add_argument("--agent_pth", type=str, default=None)
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="q_greedy",
        choices=["epsilon_greedy", "q_greedy", "softmax_sample"],
        help="Decode agent Q-values with epsilon-greedy, greedy, or softmax sampling.",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--boformer_n_layers", type=int, default=BOFORMER_N_LAYERS)
    parser.add_argument("--boformer_n_heads", type=int, default=BOFORMER_N_HEADS)
    parser.add_argument("--boformer_window_size", type=int, default=BOFORMER_WINDOW_SIZE)
    parser.add_argument("--boformer_dropout", type=float, default=BOFORMER_DROPOUT)
    parser.add_argument("--boformer_temperature", type=float, default=BOFORMER_TEMPERATURE)
    parser.add_argument("--infill", type=str, default=None)
    parser.add_argument("--compare_infill", type=str, default=None)
    parser.add_argument("--compare_algo", type=str, default=None)
    parser.add_argument("--compare_agent_pth", type=str, default=None)
    parser.add_argument("--solver", type=str, default="hybrid", choices=["moead_ego", "hybrid"])
    parser.add_argument("--hybrid_nsga3_size", type=int, default=None)
    parser.add_argument("--hybrid_moead_ego_size", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()
    args.surrogate_model = "gp"
    args.reward_lambda = None
    args.batch = 1
    args.problem = str(args.problem).upper()
    args.policy_mode_user = any(
        token == "--policy_mode" or token.startswith("--policy_mode=")
        for token in sys.argv[1:]
    )
    args.offspring_size_user = args.offspring_size is not None
    if not bool(args.policy_mode_user) and resolve_primary_policy_name(args) == "boformer":
        args.policy_mode = "softmax_sample"
    args.saea_steps = {
        "hybrid": 15,
        "moead_ego": 50,
    }[str(args.solver).lower()]
    if str(args.solver).lower() == "moead_ego" and not bool(args.offspring_size_user):
        args.offspring_size = 150
    if is_random_agent_path(args.compare_agent_pth):
        args.compare_agent_pth = None
        args.compare_random_model = True
    else:
        args.compare_random_model = False
    if args.dim is None:
        args.dim = default_problem_dim(args.problem)

    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    if args.offspring_size is None:
        args.offspring_size = 80
    if int(args.offspring_size) <= 0:
        raise ValueError(f"offspring_size must be positive, got {args.offspring_size}.")
    if args.hybrid_nsga3_size is not None:
        if int(args.hybrid_nsga3_size) < 0:
            raise ValueError(f"hybrid_nsga3_size must be non-negative, got {args.hybrid_nsga3_size}.")
    if args.hybrid_moead_ego_size is not None:
        if int(args.hybrid_moead_ego_size) < 0:
            raise ValueError(f"hybrid_moead_ego_size must be non-negative, got {args.hybrid_moead_ego_size}.")
    return args


def resolve_hybrid_branch_sizes(args: argparse.Namespace) -> tuple[int, int]:
    nsga_size_raw = getattr(args, "hybrid_nsga3_size", None)
    moead_size_raw = getattr(args, "hybrid_moead_ego_size", None)
    nsga_size = 80 if nsga_size_raw is None else int(nsga_size_raw)
    moead_size = 150 if moead_size_raw is None else int(moead_size_raw)
    return nsga_size, moead_size


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def resolve_seed(seed_value: Any) -> int:
    text = str(seed_value).strip().lower()
    if text in {"", "auto", "none"}:
        return int((time.time_ns() ^ os.getpid() ^ secrets.randbits(32)) % (2**31 - 1))
    return int(seed_value)


def fe_progress(*, init_fe: int, step: int, max_fe: int) -> float:
    max_fe_value = int(max_fe)
    if max_fe_value <= 0:
        raise ValueError(f"max_fe must be positive, got {max_fe_value}.")
    current_fe = min(int(init_fe) + int(step), max_fe_value - 1)
    return float(current_fe) / float(max_fe_value)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def resolve_agent_cls(agent_name: str):
    name = str(agent_name).strip().lower()
    if name == "disc":
        return Disc
    if name == "disc_af":
        return DiscAF
    if name == "disc_af2":
        return DiscAF2
    if name == "disc_single_dqn":
        return DiscSingleDQN
    if name == "db_saea":
        return DBSAEAAgent
    if name == "boformer":
        return BOFormer
    raise ValueError(f"Unsupported agent_name: {agent_name}")


def build_named_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray, name: str):
    surrogate_name = str(name).lower()
    if surrogate_name == "gp":
        return fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
            nu=float(getattr(args, "gp_nu", 5.0)),
        )

    raise ValueError(f"Unsupported surrogate_model: {surrogate_name}")


def build_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray):
    return build_named_surrogate(args, archive_x, archive_y, surrogate_model_name(args))


def make_nsga2_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def _generate_single_offspring_pool(
    *,
    args: argparse.Namespace,
    nsga_problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    surrogate: Any,
    step: int,
    surrogate_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solver_name = str(getattr(args, "solver", "nsga2")).lower()
    if solver_name == "moead_ego":
        result = run_surrogate_moead_ego(
            problem=nsga_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            surrogate=surrogate,
            pop_size=int(args.offspring_size) if bool(getattr(args, "offspring_size_user", False)) else None,
            saea_steps=int(args.saea_steps),
            seed=int(args.seed) + int(step),
        )
        return (
            np.asarray(result.x, dtype=np.float32),
            np.asarray(result.mean, dtype=np.float32),
            np.asarray(result.std, dtype=np.float32),
            {
                "candidate_weight_vectors": np.asarray(
                    result.weight_vectors,
                    dtype=np.float32,
                ),
                "eti": np.asarray(result.eti, dtype=np.float32),
            },
        )
    if solver_name == "hybrid":
        hybrid_nsga3_size, hybrid_moead_ego_size = resolve_hybrid_branch_sizes(args)
        hybrid_nsga3_steps, hybrid_moead_ego_steps = resolve_hybrid_branch_steps(args)
        result = run_surrogate_hybrid(
            problem=nsga_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            surrogate=surrogate,
            pop_size=int(hybrid_nsga3_size) + int(hybrid_moead_ego_size),
            saea_steps=None,
            seed=int(args.seed) + int(step),
            nsga3_pop_size=int(hybrid_nsga3_size),
            moead_ego_pop_size=int(hybrid_moead_ego_size),
            nsga3_saea_steps=int(hybrid_nsga3_steps),
            moead_ego_saea_steps=int(hybrid_moead_ego_steps),
        )
        reference_vectors = getattr(result, "weight_vectors", None)
        return (
            np.asarray(result.x, dtype=np.float32),
            np.asarray(result.mean, dtype=np.float32),
            np.asarray(result.std, dtype=np.float32),
            {
                "reference_vectors": (
                    None
                    if reference_vectors is None
                    else np.asarray(reference_vectors, dtype=np.float32)
                )
            },
        )

    raise ValueError(f"Unsupported solver: {solver_name}")


def generate_offspring_pool(
    *,
    args: argparse.Namespace,
    nsga_problem,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], dict[str, Any]]:
    solver_name = str(getattr(args, "solver", "nsga3")).lower()
    surrogate_name = "gp" if solver_name == "hybrid" else surrogate_model_name(args)
    surrogate = build_named_surrogate(args, archive_x, archive_y, surrogate_name)
    offspring_x, offspring_pred, offspring_sigma, offspring_info = _generate_single_offspring_pool(
        args=args,
        nsga_problem=nsga_problem,
        archive_x=archive_x,
        archive_y=archive_y,
        surrogate=surrogate,
        step=step,
        surrogate_name=surrogate_name,
    )
    group_indices = [np.arange(int(offspring_x.shape[0]), dtype=np.int64)]
    return offspring_x, offspring_pred, offspring_sigma, group_indices, offspring_info


def refresh_offspring_predictions(
    *,
    args: argparse.Namespace,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    offspring_x_arr = np.asarray(offspring_x, dtype=np.float32)
    if int(offspring_x_arr.shape[0]) <= 0:
        n_obj = int(np.asarray(archive_y, dtype=np.float32).shape[1])
        empty = np.empty((0, n_obj), dtype=np.float32)
        return empty, empty.copy()
    solver_name = str(getattr(args, "solver", "nsga3")).lower()
    surrogate_name = "gp" if solver_name == "hybrid" else surrogate_model_name(args)
    surrogate = build_named_surrogate(
        args,
        np.asarray(archive_x, dtype=np.float32),
        np.asarray(archive_y, dtype=np.float32),
        surrogate_name,
    )
    refreshed_mean = predict_surrogate_mean(surrogate, offspring_x_arr)
    refreshed_std = predict_surrogate_std(surrogate, offspring_x_arr)
    return (
        np.asarray(refreshed_mean, dtype=np.float32),
        np.asarray(refreshed_std, dtype=np.float32),
    )


def predict_surrogate_mean(surrogate: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(
    surrogate: Any,
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_std"):
        try:
            return np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        except NotImplementedError:
            pass
    return np.zeros((int(x_arr.shape[0]), 1), dtype=np.float32)


def build_offspring_sigma(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    surrogate: Any,
) -> np.ndarray:
    archive_y = np.asarray(archive_y, dtype=np.float32)

    sigma = predict_surrogate_std(surrogate, offspring_x)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)

    if sigma.shape[1] == archive_y.shape[1]:
        return sigma.astype(np.float32)

    archive_pred = predict_surrogate_mean(surrogate, archive_x)
    local_sigma = estimate_uncertainty(
        archive_x=archive_x,
        archive_y=archive_y,
        archive_pred=archive_pred,
        offspring_x=offspring_x,
    )
    if local_sigma.ndim == 1:
        local_sigma = local_sigma.reshape(-1, 1)
    if local_sigma.shape[1] != archive_y.shape[1]:
        local_sigma = np.repeat(local_sigma.mean(axis=1, keepdims=True), archive_y.shape[1], axis=1)
    return local_sigma.astype(np.float32)


def _pad_boformer_objective_array(values: np.ndarray, target_n_objectives: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        out = np.zeros(int(target_n_objectives), dtype=np.float32)
        out[: arr.shape[0]] = arr
        return out
    if arr.ndim == 2:
        out = np.zeros((arr.shape[0], int(target_n_objectives)), dtype=np.float32)
        out[:, : arr.shape[1]] = arr
        return out
    raise ValueError(f"Unsupported BOFormer objective array rank: {arr.ndim}.")


def _build_boformer_objective_mask(n_objectives: int, target_n_objectives: int) -> np.ndarray:
    mask = np.zeros(int(target_n_objectives), dtype=np.float32)
    mask[: int(n_objectives)] = 1.0
    return mask


def _boformer_variance_from_std(values: np.ndarray) -> np.ndarray:
    std_arr = np.asarray(values, dtype=np.float32)
    return np.square(np.maximum(std_arr, 0.0), dtype=np.float32)


def _normalize_boformer_objective_inputs(
    archive_y: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    candidate_mean_arr = np.asarray(candidate_mean, dtype=np.float32)
    candidate_std_arr = np.asarray(candidate_std, dtype=np.float32)
    stacked_y = np.vstack([archive_y_arr, candidate_mean_arr]).astype(np.float32)
    y_min = stacked_y.min(axis=0, keepdims=True)
    y_max = stacked_y.max(axis=0, keepdims=True)
    y_denom = np.clip(y_max - y_min, 1e-12, None)
    archive_y_norm = np.clip((archive_y_arr - y_min) / y_denom, 0.0, 1.0).astype(np.float32)
    candidate_mean_norm = np.clip((candidate_mean_arr - y_min) / y_denom, 0.0, 1.0).astype(np.float32)
    sigma_min = candidate_std_arr.min(axis=0, keepdims=True)
    sigma_max = candidate_std_arr.max(axis=0, keepdims=True)
    sigma_denom = np.clip(sigma_max - sigma_min, 1e-12, None)
    candidate_std_norm = np.clip((candidate_std_arr - sigma_min) / sigma_denom, 0.0, 1.0).astype(np.float32)
    return archive_y_norm, candidate_mean_norm, candidate_std_norm


def _build_boformer_obs(
    *,
    archive_y: np.ndarray,
    candidate_mean: np.ndarray,
    candidate_std: np.ndarray,
    progress: float,
    target_n_objectives: int,
) -> dict[str, np.ndarray]:
    archive_y_arr, candidate_mean_arr, candidate_std_arr = _normalize_boformer_objective_inputs(
        archive_y,
        candidate_mean,
        candidate_std,
    )
    candidate_variance_arr = _boformer_variance_from_std(candidate_std_arr)
    n_obj = int(candidate_mean_arr.shape[1])
    incumbent = archive_y_arr.min(axis=0).astype(np.float32)
    return {
        "candidate_mean": _pad_boformer_objective_array(candidate_mean_arr, target_n_objectives),
        "candidate_variance": _pad_boformer_objective_array(candidate_variance_arr, target_n_objectives),
        "best_objectives": _pad_boformer_objective_array(incumbent, target_n_objectives),
        "progress": np.asarray([float(progress)], dtype=np.float32),
        "candidate_mask": np.ones(int(candidate_mean_arr.shape[0]), dtype=bool),
        "objective_mask": _build_boformer_objective_mask(int(n_obj), int(target_n_objectives)),
    }


def _build_boformer_history_entry(
    *,
    archive_y: np.ndarray,
    selected_mean: np.ndarray,
    selected_variance_source: np.ndarray,
    target_n_objectives: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive_y_arr, selected_mean_arr, selected_std_arr = _normalize_boformer_objective_inputs(
        archive_y,
        np.asarray(selected_mean, dtype=np.float32).reshape(1, -1),
        np.asarray(selected_variance_source, dtype=np.float32).reshape(1, -1),
    )
    selected_variance = _boformer_variance_from_std(selected_std_arr.reshape(-1))
    best_arr = archive_y_arr.min(axis=0).astype(np.float32)
    return (
        _pad_boformer_objective_array(selected_mean_arr.reshape(-1), target_n_objectives),
        _pad_boformer_objective_array(selected_variance, target_n_objectives),
        _pad_boformer_objective_array(best_arr, target_n_objectives),
    )


def _build_boformer_forward_kwargs(obs: dict[str, np.ndarray], *, device: str) -> dict[str, Any]:
    return {
        "candidate_mean": torch.from_numpy(np.asarray(obs["candidate_mean"], dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "candidate_variance": torch.from_numpy(np.asarray(obs["candidate_variance"], dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "best_objectives": torch.from_numpy(np.asarray(obs["best_objectives"], dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "progress": torch.from_numpy(np.asarray(obs["progress"], dtype=np.float32).reshape(1, 1)).to(device=device, dtype=torch.float32),
        "objective_mask": torch.from_numpy(np.asarray(obs["objective_mask"], dtype=np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "candidate_mask": torch.from_numpy(np.asarray(obs["candidate_mask"], dtype=bool)).to(device=device, dtype=torch.bool).unsqueeze(0),
    }


def _build_boformer_history_kwargs(
    *,
    history_mean: list[np.ndarray],
    history_variance: list[np.ndarray],
    history_best: list[np.ndarray],
    history_progress: list[float],
    history_rewards: list[float],
    history_q_values: list[float],
    history_objective_mask: list[np.ndarray],
    device: str,
) -> dict[str, Any]:
    if not history_mean:
        return {}
    return {
        "history_mean": torch.from_numpy(np.stack(history_mean, axis=0).astype(np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "history_variance": torch.from_numpy(np.stack(history_variance, axis=0).astype(np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "history_best_objectives": torch.from_numpy(np.stack(history_best, axis=0).astype(np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "history_progress": torch.from_numpy(np.asarray(history_progress, dtype=np.float32).reshape(1, -1)).to(device=device, dtype=torch.float32),
        "history_rewards": torch.from_numpy(np.asarray(history_rewards, dtype=np.float32).reshape(1, -1)).to(device=device, dtype=torch.float32),
        "history_q_values": torch.from_numpy(np.asarray(history_q_values, dtype=np.float32).reshape(1, -1)).to(device=device, dtype=torch.float32),
        "history_objective_mask": torch.from_numpy(np.stack(history_objective_mask, axis=0).astype(np.float32)).to(device=device, dtype=torch.float32).unsqueeze(0),
        "history_mask": torch.ones(1, len(history_mean), dtype=torch.bool, device=device),
    }


def _infer_boformer_init_kwargs(
    args: argparse.Namespace,
    *,
    state_dict: dict[str, torch.Tensor] | None,
    n_objectives: int,
) -> dict[str, Any]:
    hidden_dim = int(args.hidden_dim)
    objective_count = int(n_objectives)
    n_layers = int(getattr(args, "boformer_n_layers", BOFORMER_N_LAYERS))
    window_size = int(getattr(args, "boformer_window_size", BOFORMER_WINDOW_SIZE))
    if state_dict is not None:
        obs_weight = state_dict.get("observation_action_weight")
        if obs_weight is not None:
            hidden_dim = int(obs_weight.shape[0])
            if int(obs_weight.ndim) == 3:
                objective_count = int(obs_weight.shape[1])
            elif int(obs_weight.ndim) == 2 and int(obs_weight.shape[1]) >= 4:
                objective_count = int((int(obs_weight.shape[1]) - 1) // 3)
        window_weight = state_dict.get("transformer.wpe.weight")
        if window_weight is not None and int(window_weight.ndim) == 2:
            window_size = max(1, int(window_weight.shape[0]))
        layer_indices: set[int] = set()
        for key in state_dict.keys():
            match = re.match(r"transformer\.h\.(\d+)\.", str(key))
            if match:
                layer_indices.add(int(match.group(1)))
        if layer_indices:
            n_layers = max(layer_indices) + 1
    return {
        "n_objectives": int(objective_count),
        "hidden_dim": int(hidden_dim),
        "n_layers": int(n_layers),
        "n_heads": int(getattr(args, "boformer_n_heads", BOFORMER_N_HEADS)),
        "window_size": int(window_size),
        "dropout": float(getattr(args, "boformer_dropout", BOFORMER_DROPOUT)),
        "temperature": float(getattr(args, "boformer_temperature", BOFORMER_TEMPERATURE)),
    }


def _infer_db_saea_init_kwargs(
    args: argparse.Namespace,
    *,
    state_dict: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    hidden_dim = int(getattr(args, "hidden_dim", 16))
    ff_dim = int(getattr(args, "ff_dim", 256))
    decoder_hidden_dim = 16
    if state_dict is not None:
        w_true = state_dict.get("W_true.weight")
        if w_true is not None and int(w_true.ndim) == 2:
            hidden_dim = int(w_true.shape[0])
        ff0 = state_dict.get("encoder_true.cross_individual.ff.0.weight")
        if ff0 is not None and int(ff0.ndim) == 2:
            ff_dim = int(ff0.shape[0])
        decoder_w = state_dict.get("q_decoder.advantage_head.1.weight")
        if decoder_w is not None and int(decoder_w.ndim) == 2:
            decoder_hidden_dim = int(decoder_w.shape[0])
    return {
        "hidden_dim": int(hidden_dim),
        "n_heads": 1,
        "ff_dim": int(ff_dim),
        "dropout": float(getattr(args, "dropout", 0.0)),
        "logit_scale": float(getattr(args, "logit_scale", 5.0)),
        "decoder_hidden_dim": int(decoder_hidden_dim),
    }


def preselect_boformer_batch(
    *,
    args: argparse.Namespace,
    boformer: BOFormer,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    ref_point: np.ndarray,
    reward_scheme_id: int,
    true_pareto_hv: float | None,
    start_step: int,
    n_select: int,
    history_mean: list[np.ndarray],
    history_variance: list[np.ndarray],
    history_best: list[np.ndarray],
    history_progress: list[float],
    history_rewards: list[float],
    history_q_values: list[float],
    history_objective_mask: list[np.ndarray],
) -> dict[str, Any]:
    remaining_x = np.asarray(offspring_x, dtype=np.float32).copy()
    current_mean = np.asarray(offspring_pred, dtype=np.float32).copy()
    current_std = np.asarray(offspring_sigma, dtype=np.float32).copy()
    remaining_indices = np.arange(int(remaining_x.shape[0]), dtype=np.int64)
    temp_archive_x = np.asarray(archive_x, dtype=np.float32).copy()
    temp_archive_y = np.asarray(archive_y, dtype=np.float32).copy()

    pseudo_history_mean = [np.asarray(item, dtype=np.float32).copy() for item in history_mean]
    pseudo_history_variance = [np.asarray(item, dtype=np.float32).copy() for item in history_variance]
    pseudo_history_best = [np.asarray(item, dtype=np.float32).copy() for item in history_best]
    pseudo_history_progress = [float(item) for item in history_progress]
    pseudo_history_rewards = [float(item) for item in history_rewards]
    pseudo_history_q_values = [float(item) for item in history_q_values]
    pseudo_history_objective_mask = [np.asarray(item, dtype=np.float32).copy() for item in history_objective_mask]

    selected_indices: list[int] = []
    selected_x_parts: list[np.ndarray] = []
    selected_mean_parts: list[np.ndarray] = []
    selected_variance_source_parts: list[np.ndarray] = []
    selected_best_parts: list[np.ndarray] = []
    selected_progress_parts: list[float] = []
    selected_q_parts: list[float] = []
    selected_objective_mask_parts: list[np.ndarray] = []

    target_n_objectives = int(boformer.n_objectives)

    for local_step in range(min(int(n_select), int(remaining_x.shape[0]))):
        progress = fe_progress(
            init_fe=int(args.init_fe),
            step=int(start_step) + int(local_step),
            max_fe=int(args.max_fe),
        )
        obs = _build_boformer_obs(
            archive_y=temp_archive_y,
            candidate_mean=current_mean,
            candidate_std=current_std,
            progress=progress,
            target_n_objectives=target_n_objectives,
        )
        forward_kwargs = _build_boformer_forward_kwargs(obs, device=str(args.device))
        forward_kwargs.update(
            _build_boformer_history_kwargs(
                history_mean=pseudo_history_mean,
                history_variance=pseudo_history_variance,
                history_best=pseudo_history_best,
                history_progress=pseudo_history_progress,
                history_rewards=pseudo_history_rewards,
                history_q_values=pseudo_history_q_values,
                history_objective_mask=pseudo_history_objective_mask,
                device=str(args.device),
            )
        )
        with torch.no_grad():
            out = boformer(
                **forward_kwargs,
                decode_type=str(getattr(args, "policy_mode", "softmax_sample")),
                epsilon=0.0 if str(getattr(args, "policy_mode", "softmax_sample")) != "epsilon_greedy" else 0.05,
            )
        selected_local_idx = int(out["ranking"].reshape(out["ranking"].shape[0], -1)[0, 0].item())
        selected_q = float(out["q_values"][0, int(selected_local_idx)].detach().cpu().item())
        selected_x = np.asarray(remaining_x[selected_local_idx : selected_local_idx + 1], dtype=np.float32)
        selected_mean = np.asarray(current_mean[selected_local_idx], dtype=np.float32)
        selected_variance_source = np.asarray(current_std[selected_local_idx], dtype=np.float32)
        history_mean_entry, history_variance_entry, selected_best = _build_boformer_history_entry(
            archive_y=temp_archive_y,
            selected_mean=selected_mean,
            selected_variance_source=selected_variance_source,
            target_n_objectives=target_n_objectives,
        )
        selected_objective_mask = np.asarray(obs["objective_mask"], dtype=np.float32)

        pseudo_reward = compute_test_reward(
            reward_scheme_id=int(reward_scheme_id),
            previous_front=pareto_front(np.asarray(temp_archive_y, dtype=np.float32)),
            selected_objectives=selected_mean.reshape(1, -1),
            ref_point=np.asarray(ref_point, dtype=np.float32),
            reward_lambda=float(args.reward_lambda),
            true_pareto_hv=true_pareto_hv,
        )

        pseudo_history_mean.append(history_mean_entry.copy())
        pseudo_history_variance.append(history_variance_entry.copy())
        pseudo_history_best.append(selected_best.copy())
        pseudo_history_progress.append(float(progress))
        pseudo_history_rewards.append(float(pseudo_reward))
        pseudo_history_q_values.append(float(selected_q))
        pseudo_history_objective_mask.append(selected_objective_mask.copy())

        selected_indices.append(int(remaining_indices[selected_local_idx]))
        selected_x_parts.append(selected_x.copy())
        selected_mean_parts.append(selected_mean.copy())
        selected_variance_source_parts.append(selected_variance_source.copy())
        selected_best_parts.append(selected_best.copy())
        selected_progress_parts.append(float(progress))
        selected_q_parts.append(float(selected_q))
        selected_objective_mask_parts.append(selected_objective_mask.copy())

        temp_archive_x = np.vstack([temp_archive_x, selected_x]).astype(np.float32)
        temp_archive_y = np.vstack([temp_archive_y, selected_mean.reshape(1, -1)]).astype(np.float32)

        keep_mask = np.ones(int(remaining_x.shape[0]), dtype=bool)
        keep_mask[int(selected_local_idx)] = False
        remaining_x = remaining_x[keep_mask]
        remaining_indices = remaining_indices[keep_mask]
        current_mean = current_mean[keep_mask]
        current_std = current_std[keep_mask]
        if int(remaining_x.shape[0]) <= 0:
            break

        surrogate_args = argparse.Namespace(**vars(args))
        surrogate_args.seed = int(args.seed) + int(start_step) + int(local_step) + 1
        temp_surrogate = build_named_surrogate(surrogate_args, temp_archive_x, temp_archive_y, "gp")
        current_mean = predict_surrogate_mean(temp_surrogate, remaining_x)
        current_std = build_offspring_sigma(
            archive_x=temp_archive_x,
            archive_y=temp_archive_y,
            offspring_x=remaining_x,
            surrogate=temp_surrogate,
        )

    if selected_x_parts:
        selected_x_arr = np.vstack(selected_x_parts).astype(np.float32)
        selected_mean_arr = np.vstack([item.reshape(1, -1) for item in selected_mean_parts]).astype(np.float32)
        selected_variance_source_arr = np.vstack([item.reshape(1, -1) for item in selected_variance_source_parts]).astype(np.float32)
        selected_best_arr = np.vstack([item.reshape(1, -1) for item in selected_best_parts]).astype(np.float32)
        selected_progress_arr = np.asarray(selected_progress_parts, dtype=np.float32)
        selected_q_arr = np.asarray(selected_q_parts, dtype=np.float32)
        selected_mask_arr = np.vstack([item.reshape(1, -1) for item in selected_objective_mask_parts]).astype(np.float32)
    else:
        selected_x_arr = np.zeros((0, int(offspring_x.shape[1])), dtype=np.float32)
        selected_mean_arr = np.zeros((0, int(offspring_pred.shape[1])), dtype=np.float32)
        selected_variance_source_arr = np.zeros((0, int(offspring_sigma.shape[1])), dtype=np.float32)
        selected_best_arr = np.zeros((0, target_n_objectives), dtype=np.float32)
        selected_progress_arr = np.zeros((0,), dtype=np.float32)
        selected_q_arr = np.zeros((0,), dtype=np.float32)
        selected_mask_arr = np.zeros((0, target_n_objectives), dtype=np.float32)

    return {
        "x": selected_x_arr,
        "mean": selected_mean_arr,
        "variance_source": selected_variance_source_arr,
        "indices": np.asarray(selected_indices, dtype=np.int64),
        "best": selected_best_arr,
        "progress": selected_progress_arr,
        "q_values": selected_q_arr,
        "objective_mask": selected_mask_arr,
        "cursor": 0,
    }


def build_disc(
    args: argparse.Namespace,
    *,
    map_location: str,
    agent_name: str = "disc",
    n_objectives: int | None = None,
    state_dict_override: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.nn.Module, dict[str, float]]:
    agent_cls = resolve_agent_cls(agent_name)
    agent_key = str(agent_name).strip().lower()
    torch_load_sec = 0.0
    load_state_dict_sec = 0.0
    state_dict = None if state_dict_override is None else state_dict_override

    if state_dict is None and args.agent_pth and not bool(args.random_model):
        torch_load_started_at = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            state = torch.load(args.agent_pth, map_location=map_location)
        torch_load_sec = time.perf_counter() - torch_load_started_at
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    state_dict = upgrade_legacy_observation_action_weight(state_dict)

    if agent_key == "boformer":
        if n_objectives is None and state_dict is None:
            raise ValueError("BOFormer build requires n_objectives when no checkpoint is provided.")
        agent_kwargs = _infer_boformer_init_kwargs(
            args,
            state_dict=state_dict,
            n_objectives=int(n_objectives if n_objectives is not None else 1),
        )
    elif agent_key == "db_saea":
        agent_kwargs = _infer_db_saea_init_kwargs(
            args,
            state_dict=state_dict,
        )
    else:
        agent_kwargs = {
            "hidden_dim": int(args.hidden_dim),
            "n_heads": int(args.n_heads),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "logit_scale": float(args.logit_scale),
        }
    model_to_device_started_at = time.perf_counter()
    disc = agent_cls(**agent_kwargs).to(map_location)
    model_to_device_sec = time.perf_counter() - model_to_device_started_at
    disc.eval()

    if state_dict is not None:
        load_state_dict_started_at = time.perf_counter()
        disc.load_state_dict(state_dict, strict=True)
        load_state_dict_sec = time.perf_counter() - load_state_dict_started_at

    return disc, {
        "model_to_device_sec": float(model_to_device_sec),
        "torch_load_sec": float(torch_load_sec),
        "load_state_dict_sec": float(load_state_dict_sec),
    }


def prediction_history_kwargs(agent, y_true_history, sigma_true_history):
    if not bool(getattr(agent, "uses_prediction_history", False)):
        return {}
    return {
        "y_true_history": y_true_history,
        "sigma_true_history": sigma_true_history,
    }


def build_prediction_history_kwargs(agent, y_history: np.ndarray, sigma_history: np.ndarray, *, device: str):
    if not bool(getattr(agent, "uses_prediction_history", False)):
        return {}
    return {
        "y_true_history": torch.from_numpy(np.asarray(y_history, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
        ),
        "sigma_true_history": torch.from_numpy(np.asarray(sigma_history, dtype=np.float32)).to(
            device=device,
            dtype=torch.float32,
        ),
    }


def build_trajectory_history_kwargs(
    agent,
    state_embeddings: list[torch.Tensor],
    timesteps: list[float],
    q_values: list[float],
    *,
    device: str,
) -> dict[str, torch.Tensor]:
    if not bool(getattr(agent, "uses_trajectory_history", False)) or not state_embeddings:
        return {}
    embeddings = torch.stack(state_embeddings, dim=0).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    timestep_tensor = torch.tensor(timesteps, device=device, dtype=torch.float32).unsqueeze(0)
    q_tensor = torch.tensor(q_values, device=device, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, len(state_embeddings), device=device, dtype=torch.bool)
    return {
        "history_state_embeddings": embeddings,
        "history_timesteps": timestep_tensor,
        "history_q_values": q_tensor,
        "history_mask": mask,
    }


def append_trajectory_history(
    agent,
    output: dict[str, torch.Tensor],
    selected_idx: int,
    progress: float,
    state_embeddings: list[torch.Tensor],
    timesteps: list[float],
    q_values: list[float],
) -> None:
    if not bool(getattr(agent, "uses_trajectory_history", False)):
        return
    embeddings = output.get("state_action_embeddings")
    scores = output.get("q_values")
    if embeddings is None or scores is None:
        raise RuntimeError("Trajectory-aware agent must return state_action_embeddings and q_values.")
    state_embeddings.append(embeddings[0, int(selected_idx)].detach().cpu())
    timesteps.append(float(progress))
    q_values.append(float(scores[0, int(selected_idx)].detach().cpu().item()))


def init_archive_prediction_history(archive_y: np.ndarray, *, history_steps: int = 40) -> tuple[np.ndarray, np.ndarray]:
    archive_y_arr = np.asarray(archive_y, dtype=np.float32)
    y_history = np.repeat(archive_y_arr[:, None, :], int(history_steps), axis=1).astype(np.float32)
    sigma_history = np.zeros_like(y_history, dtype=np.float32)
    return y_history, sigma_history


def append_archive_prediction_history(
    y_history: np.ndarray,
    sigma_history: np.ndarray,
    *,
    selected_pred: np.ndarray,
    selected_sigma: np.ndarray,
    selected_true: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    history_steps = int(np.asarray(y_history).shape[1])
    pred = np.asarray(selected_pred, dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(selected_sigma, dtype=np.float32).reshape(1, -1)
    true = np.asarray(selected_true, dtype=np.float32).reshape(1, -1)
    new_y_history = np.repeat(pred[:, None, :], history_steps, axis=1).astype(np.float32)
    new_sigma_history = np.repeat(sigma[:, None, :], history_steps, axis=1).astype(np.float32)
    eval_step = min(max(int(step), 0), history_steps - 1)
    new_y_history[:, eval_step:, :] = true[:, None, :]
    new_sigma_history[:, eval_step:, :] = 0.0
    return (
        np.vstack([np.asarray(y_history, dtype=np.float32), new_y_history]).astype(np.float32),
        np.vstack([np.asarray(sigma_history, dtype=np.float32), new_sigma_history]).astype(np.float32),
    )


@dataclass
class StepRecord:
    step: int
    fe: int
    selected_index: int
    selected_x: list[float]
    surrogate_y: list[float]
    true_y: list[float]
    reward: float
    hv: float
    archive_size: int


def remove_offspring_candidate(
    offspring_x: np.ndarray,
    offspring_pred: np.ndarray,
    offspring_sigma: np.ndarray,
    offspring_groups: list[np.ndarray],
    selected_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    n_candidates = int(np.asarray(offspring_x).shape[0])
    keep_mask = np.ones(n_candidates, dtype=bool)
    keep_mask[int(selected_idx)] = False
    keep_idx = np.where(keep_mask)[0].astype(np.int64)
    old_to_new = np.full(n_candidates, -1, dtype=np.int64)
    old_to_new[keep_idx] = np.arange(int(keep_idx.shape[0]), dtype=np.int64)

    new_groups: list[np.ndarray] = []
    for group in offspring_groups:
        group_arr = np.asarray(group, dtype=np.int64).reshape(-1)
        group_arr = group_arr[group_arr != int(selected_idx)]
        remapped = old_to_new[group_arr]
        remapped = remapped[remapped >= 0]
        if remapped.size > 0:
            new_groups.append(remapped.astype(np.int64))
    if len(new_groups) == 0 and keep_idx.size > 0:
        new_groups = [np.arange(int(keep_idx.shape[0]), dtype=np.int64)]

    return (
        np.asarray(offspring_x, dtype=np.float32)[keep_idx],
        np.asarray(offspring_pred, dtype=np.float32)[keep_idx],
        np.asarray(offspring_sigma, dtype=np.float32)[keep_idx],
        new_groups,
    )


def remove_offspring_info_candidate(
    offspring_info: dict[str, Any],
    *,
    selected_idx: int,
    n_candidates: int,
) -> dict[str, Any]:
    """Keep candidate-aligned solver metadata synchronized with a shrinking pool."""
    updated = dict(offspring_info)
    for key in ("candidate_weight_vectors", "eti"):
        value = updated.get(key)
        if value is None:
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and int(array.shape[0]) == int(n_candidates):
            updated[key] = np.delete(array, int(selected_idx), axis=0)
    return updated


def run_policy_rollout_batched(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    archive_x_init: np.ndarray,
    archive_y_init: np.ndarray,
    policy_name: str,
    disc: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    make_plot: bool = True,
    logger=print,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_x = np.asarray(archive_x_init, dtype=np.float32).copy()
    archive_y = np.asarray(archive_y_init, dtype=np.float32).copy()
    archive_y_history, archive_sigma_history = init_archive_prediction_history(archive_y, history_steps=40)
    n_evo_steps = int(args.max_fe) - int(args.init_fe)
    batch_size = max(1, int(getattr(args, "batch", 1)))
    fe_history = [int(args.init_fe)]
    hv_history = [float(hypervolume(archive_y, ref_point))]
    history: list[StepRecord] = []
    trajectory_state_embeddings: list[torch.Tensor] = []
    trajectory_timesteps: list[float] = []
    trajectory_q_values: list[float] = []
    boformer_history_mean: list[np.ndarray] = []
    boformer_history_variance: list[np.ndarray] = []
    boformer_history_best: list[np.ndarray] = []
    boformer_history_progress: list[float] = []
    boformer_history_rewards: list[float] = []
    boformer_history_q_values: list[float] = []
    boformer_history_objective_mask: list[np.ndarray] = []
    step_rewards: list[float] = []
    if infill_criterion is not None and hasattr(infill_criterion, "reset"):
        infill_criterion.reset()

    prefix = format_policy_prefix(policy_name)
    logger(f"{prefix}iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.12f}")

    step = 0
    pool_id = 0
    policy_key = str(policy_name).lower()
    while step < n_evo_steps:
        offspring_x, offspring_pred, offspring_sigma, offspring_groups, offspring_info = generate_offspring_pool(
            args=args,
            nsga_problem=nsga2_problem,
            archive_x=archive_x,
            archive_y=archive_y,
            step=pool_id,
        )
        pool_id += 1
        pool_evals = min(batch_size, n_evo_steps - step)

        for _ in range(pool_evals):
            if step >= n_evo_steps or int(offspring_x.shape[0]) <= 0:
                break
            boformer_selected_best = None
            boformer_selected_progress = None
            boformer_selected_q = None
            boformer_selected_objective_mask = None
            if policy_key in {"disc", "disc_af", "disc_af2", "disc_single_dqn"}:
                if disc is None:
                    raise ValueError("DISC rollout requires a built disc model.")
                progress = fe_progress(
                    init_fe=int(args.init_fe),
                    step=int(step),
                    max_fe=int(args.max_fe),
                )
                history_kwargs = build_prediction_history_kwargs(
                    disc,
                    archive_y_history,
                    archive_sigma_history,
                    device=str(args.device),
                )
                history_kwargs.update(
                    build_trajectory_history_kwargs(
                        disc,
                        trajectory_state_embeddings,
                        trajectory_timesteps,
                        trajectory_q_values,
                        device=str(args.device),
                    )
                )
                with torch.no_grad():
                    out = disc(
                        x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                        y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                        x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                        y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                        sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                        progress=progress,
                        lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                        upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                        decode_type=str(args.policy_mode),
                        epsilon=0.05,
                        **history_kwargs,
                    )
                selected_idx = int(out["ranking"].reshape(out["ranking"].shape[0], -1)[0, 0].item())
                append_trajectory_history(
                    disc,
                    out,
                    selected_idx,
                    progress,
                    trajectory_state_embeddings,
                    trajectory_timesteps,
                    trajectory_q_values,
                )
            elif policy_key == "boformer":
                if disc is None:
                    raise ValueError("BOFormer rollout requires a built boformer model.")
                boformer_one_step = preselect_boformer_batch(
                    args=args,
                    boformer=disc,
                    archive_x=archive_x,
                    archive_y=archive_y,
                    offspring_x=offspring_x,
                    offspring_pred=offspring_pred,
                    offspring_sigma=offspring_sigma,
                    ref_point=ref_point,
                    reward_scheme_id=int(reward_scheme_id),
                    true_pareto_hv=true_pareto_hv,
                    start_step=int(step),
                    n_select=1,
                    history_mean=boformer_history_mean,
                    history_variance=boformer_history_variance,
                    history_best=boformer_history_best,
                    history_progress=boformer_history_progress,
                    history_rewards=boformer_history_rewards,
                    history_q_values=boformer_history_q_values,
                    history_objective_mask=boformer_history_objective_mask,
                )
                selected_idx = int(np.asarray(boformer_one_step["indices"], dtype=np.int64).reshape(-1)[0])
                boformer_selected_best = np.asarray(boformer_one_step["best"][0], dtype=np.float32)
                boformer_selected_progress = float(boformer_one_step["progress"][0])
                boformer_selected_q = float(boformer_one_step["q_values"][0])
                boformer_selected_objective_mask = np.asarray(boformer_one_step["objective_mask"][0], dtype=np.float32)
            elif infill_criterion is not None:
                selected_idx, _ = _call_infill_select_index(
                    infill_criterion,
                    **_make_infill_selection_kwargs(
                        criterion=infill_criterion,
                        archive_y=archive_y,
                        candidate_mean=offspring_pred,
                        candidate_std=offspring_sigma,
                        seed=int(args.seed) + step,
                        offspring_info=offspring_info,
                        progress_ratio=fe_progress(
                            init_fe=int(args.init_fe),
                            step=int(step),
                            max_fe=int(args.max_fe),
                        ),
                    ),
                )
            else:
                raise ValueError(f"Unsupported batched policy_name: {policy_name}")

            previous_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
            selected_x = offspring_x[selected_idx : selected_idx + 1]
            selected_pred = offspring_pred[selected_idx]
            selected_sigma = offspring_sigma[selected_idx]
            selected_true = np.asarray(problem.evaluate(selected_x), dtype=np.float32)
            step_reward = compute_test_reward(
                reward_scheme_id=int(reward_scheme_id),
                previous_front=previous_front,
                selected_objectives=selected_true,
                ref_point=ref_point,
                reward_lambda=float(args.reward_lambda),
                true_pareto_hv=true_pareto_hv,
            )

            archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
            archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
            archive_y_history, archive_sigma_history = append_archive_prediction_history(
                archive_y_history,
                archive_sigma_history,
                selected_pred=selected_pred,
                selected_sigma=selected_sigma,
                selected_true=selected_true,
                step=step,
            )
            step += 1
            hv = hypervolume(archive_y, ref_point)
            fe = int(args.init_fe) + step
            front_size = int(pareto_front(archive_y).shape[0])
            fe_history.append(fe)
            hv_history.append(float(hv))

            record = StepRecord(
                step=step,
                fe=fe,
                selected_index=int(selected_idx),
                selected_x=selected_x.reshape(-1).astype(float).tolist(),
                surrogate_y=selected_pred.astype(float).tolist(),
                true_y=selected_true.reshape(-1).astype(float).tolist(),
                reward=step_reward,
                hv=float(hv),
                archive_size=int(archive_x.shape[0]),
            )
            history.append(record)
            step_rewards.append(step_reward)
            if policy_key == "boformer":
                target_n_objectives = int(getattr(disc, "n_objectives", selected_pred.shape[0]))
                history_mean_entry, history_variance_entry, history_best_entry = _build_boformer_history_entry(
                    archive_y=previous_front,
                    selected_mean=selected_pred,
                    selected_variance_source=selected_sigma,
                    target_n_objectives=target_n_objectives,
                )
                best_padded = history_best_entry if boformer_selected_best is None else np.asarray(boformer_selected_best, dtype=np.float32)
                progress_value = (
                    fe_progress(
                        init_fe=int(args.init_fe),
                        step=int(step) - 1,
                        max_fe=int(args.max_fe),
                    )
                    if boformer_selected_progress is None
                    else float(boformer_selected_progress)
                )
                objective_mask_value = (
                    _build_boformer_objective_mask(int(selected_pred.shape[0]), target_n_objectives)
                    if boformer_selected_objective_mask is None
                    else np.asarray(boformer_selected_objective_mask, dtype=np.float32)
                )
                q_value_for_history = 0.0 if boformer_selected_q is None else float(boformer_selected_q)
                boformer_history_mean.append(history_mean_entry)
                boformer_history_variance.append(history_variance_entry)
                boformer_history_best.append(np.asarray(best_padded, dtype=np.float32))
                boformer_history_progress.append(float(progress_value))
                boformer_history_rewards.append(float(step_reward))
                boformer_history_q_values.append(float(q_value_for_history))
                boformer_history_objective_mask.append(np.asarray(objective_mask_value, dtype=np.float32))

            log_line = f"{prefix}iter {record.step} | front = {front_size} | HV = {record.hv:.12f}"
            log_line += f" | reward = {record.reward:.6f}"
            logger(log_line)

            old_pool_size = int(offspring_x.shape[0])
            offspring_x, offspring_pred, offspring_sigma, offspring_groups = remove_offspring_candidate(
                offspring_x,
                offspring_pred,
                offspring_sigma,
                offspring_groups,
                selected_idx=int(selected_idx),
            )
            offspring_info = remove_offspring_info_candidate(
                offspring_info,
                selected_idx=int(selected_idx),
                n_candidates=old_pool_size,
            )
            if (
                policy_key in {"disc", "disc_af", "disc_af2", "disc_single_dqn", "boformer"}
                and int(offspring_x.shape[0]) > 0
            ):
                offspring_pred, offspring_sigma = refresh_offspring_predictions(
                    args=args,
                    archive_x=archive_x,
                    archive_y=archive_y,
                    offspring_x=offspring_x,
                )

    final_front = pareto_front(archive_y)
    plot_path = None
    npy_paths = None
    if make_plot:
        plot_path = plot_results(
            args=args,
            fe_history=fe_history,
            hv_history=hv_history,
            archive_y=archive_y,
            true_pareto=true_pareto,
        )
        npy_paths = save_npy_outputs(
            args=args,
            archive_x=archive_x,
            archive_y=archive_y,
            final_front=final_front,
            fe_history=fe_history,
            hv_history=hv_history,
        )

    summary = {
        "problem": args.problem,
        "dim": int(args.dim),
        "seed": int(args.seed),
        "max_fe": int(args.max_fe),
        "init_fe": int(args.init_fe),
        "evolution_fe": n_evo_steps,
        "batch": batch_size,
        "surrogate_model": surrogate_model_name(args),
        "candidate_solver": str(getattr(args, "solver", "nsga2")).lower(),
        "reward_lambda": float(args.reward_lambda),
        "reward_scheme": int(reward_scheme_id),
        "agent_name": args.agent_name,
        "policy_name": policy_key,
        "agent_pth": args.agent_pth,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": float(hypervolume(archive_y, ref_point)),
        "mean_reward_40_steps": float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0,
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "plot_path": plot_path,
        "npy_paths": npy_paths,
        "history": [asdict(item) for item in history],
        "fe_history": fe_history,
        "hv_history": hv_history,
    }
    return summary, archive_y


def run_policy_rollout(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    archive_x_init: np.ndarray,
    archive_y_init: np.ndarray,
    policy_name: str,
    disc: Any | None = None,
    db_saea_agent: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    make_plot: bool = True,
    logger=print,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    batch_size = max(1, int(getattr(args, "batch", 1)))
    non_batched_policies = {"db_saea", "moead_ego"}
    if batch_size > 1 and str(policy_name).lower() not in non_batched_policies:
        return run_policy_rollout_batched(
            args=args,
            problem=problem,
            nsga2_problem=nsga2_problem,
            ref_point=ref_point,
            true_pareto=true_pareto,
            archive_x_init=archive_x_init,
            archive_y_init=archive_y_init,
            policy_name=policy_name,
            disc=disc,
            infill_criterion=infill_criterion,
            compare_mode=compare_mode,
            make_plot=make_plot,
            logger=logger,
            reward_scheme_id=reward_scheme_id,
            true_pareto_hv=true_pareto_hv,
        )

    archive_x = np.asarray(archive_x_init, dtype=np.float32).copy()
    archive_y = np.asarray(archive_y_init, dtype=np.float32).copy()
    archive_y_history, archive_sigma_history = init_archive_prediction_history(archive_y, history_steps=40)
    n_evo_steps = int(args.max_fe) - int(args.init_fe)
    fe_history = [int(args.init_fe)]
    hv_history = [float(hypervolume(archive_y, ref_point))]
    history: list[StepRecord] = []
    trajectory_state_embeddings: list[torch.Tensor] = []
    trajectory_timesteps: list[float] = []
    trajectory_q_values: list[float] = []
    boformer_history_mean: list[np.ndarray] = []
    boformer_history_variance: list[np.ndarray] = []
    boformer_history_best: list[np.ndarray] = []
    boformer_history_progress: list[float] = []
    boformer_history_rewards: list[float] = []
    boformer_history_q_values: list[float] = []
    boformer_history_objective_mask: list[np.ndarray] = []
    step_rewards: list[float] = []
    boformer_batch_cache: dict[str, Any] | None = None
    if infill_criterion is not None and hasattr(infill_criterion, "reset"):
        infill_criterion.reset()

    prefix = format_policy_prefix(policy_name)
    logger(f"{prefix}iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.12f}")

    for step in range(n_evo_steps):
        policy_key = policy_name.lower()
        offspring_info: dict[str, Any] = {}
        if policy_key == "moead_ego":
            offspring_x, offspring_pred, offspring_sigma, moead_info = propose_moead_ego_candidates(
                problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                pop_size=int(args.offspring_size) if bool(getattr(args, "offspring_size_user", False)) else None,
                moead_gen=int(args.saea_steps),
                batch_size=1,
                gp_restarts=3,
                seed=int(args.seed) + int(step),
            )
            offspring_groups = [np.arange(int(offspring_x.shape[0]), dtype=np.int64)]
            offspring_info = {
                "reference_vectors": None if moead_info is None else moead_info.get("reference_vectors")
            }
        elif policy_key == "boformer" and boformer_batch_cache is not None and int(boformer_batch_cache["cursor"]) < int(boformer_batch_cache["x"].shape[0]):
            cursor = int(boformer_batch_cache["cursor"])
            offspring_x = np.asarray(boformer_batch_cache["x"][cursor : cursor + 1], dtype=np.float32)
            offspring_pred = np.asarray(boformer_batch_cache["mean"][cursor : cursor + 1], dtype=np.float32)
            offspring_sigma = np.asarray(boformer_batch_cache["variance_source"][cursor : cursor + 1], dtype=np.float32)
            offspring_groups = [np.arange(int(offspring_x.shape[0]), dtype=np.int64)]
            offspring_info = {}
        else:
            raw_offspring_x, raw_offspring_pred, raw_offspring_sigma, raw_offspring_groups, raw_offspring_info = generate_offspring_pool(
                args=args,
                nsga_problem=nsga2_problem,
                archive_x=archive_x,
                archive_y=archive_y,
                step=step,
            )
            offspring_x = raw_offspring_x
            offspring_pred = raw_offspring_pred
            offspring_sigma = raw_offspring_sigma
            offspring_groups = raw_offspring_groups
            offspring_info = raw_offspring_info
            if policy_key == "boformer":
                if disc is None:
                    raise ValueError("BOFormer rollout requires a built boformer model.")
                boformer_batch_cache = preselect_boformer_batch(
                    args=args,
                    boformer=disc,
                    archive_x=archive_x,
                    archive_y=archive_y,
                    offspring_x=raw_offspring_x,
                    offspring_pred=raw_offspring_pred,
                    offspring_sigma=raw_offspring_sigma,
                    ref_point=ref_point,
                    reward_scheme_id=int(reward_scheme_id),
                    true_pareto_hv=true_pareto_hv,
                    start_step=int(step),
                    n_select=1,
                    history_mean=boformer_history_mean,
                    history_variance=boformer_history_variance,
                    history_best=boformer_history_best,
                    history_progress=boformer_history_progress,
                    history_rewards=boformer_history_rewards,
                    history_q_values=boformer_history_q_values,
                    history_objective_mask=boformer_history_objective_mask,
                )
                offspring_x = np.asarray(boformer_batch_cache["x"][0:1], dtype=np.float32)
                offspring_pred = np.asarray(boformer_batch_cache["mean"][0:1], dtype=np.float32)
                offspring_sigma = np.asarray(boformer_batch_cache["variance_source"][0:1], dtype=np.float32)
                offspring_groups = [np.arange(int(offspring_x.shape[0]), dtype=np.int64)]
        if policy_name.lower() in {"disc", "disc_af", "disc_af2", "disc_single_dqn"}:
            if disc is None:
                raise ValueError("DISC rollout requires a built disc model.")
            progress = fe_progress(
                init_fe=int(args.init_fe),
                step=int(step),
                max_fe=int(args.max_fe),
            )
            history_kwargs = build_prediction_history_kwargs(
                disc,
                archive_y_history,
                archive_sigma_history,
                device=str(args.device),
            )
            history_kwargs.update(
                build_trajectory_history_kwargs(
                    disc,
                    trajectory_state_embeddings,
                    trajectory_timesteps,
                    trajectory_q_values,
                    device=str(args.device),
                )
            )
            with torch.no_grad():
                out = disc(
                    x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                    y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                    x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                    y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                    sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                    progress=progress,
                    lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                    upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                    decode_type=str(args.policy_mode),
                    epsilon=0.05,
                    **history_kwargs,
                )
            selected_idx = int(out["ranking"].reshape(out["ranking"].shape[0], -1)[0, 0].item())
            append_trajectory_history(
                disc,
                out,
                selected_idx,
                progress,
                trajectory_state_embeddings,
                trajectory_timesteps,
                trajectory_q_values,
            )
        elif policy_name.lower() == "db_saea":
            if db_saea_agent is None:
                raise ValueError("DB-SAEA rollout requires a built db_saea model.")
            max_regenerate_attempts = 3
            regenerate_attempts = 0
            while True:
                progress = fe_progress(
                    init_fe=int(args.init_fe),
                    step=int(step),
                    max_fe=int(args.max_fe),
                )
                with torch.no_grad():
                    out = db_saea_agent(
                        x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                        y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                        x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                        y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                        sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                        progress=progress,
                        lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                        upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                        decode_type=str(args.policy_mode),
                        epsilon=0.05,
                    )
                strategy_action = int(out["action"].reshape(-1)[0].item())
                if strategy_action == 0 and regenerate_attempts >= max_regenerate_attempts:
                    strategy_action = 1 + int(torch.argmax(out["q_values"].reshape(-1)[1:]).item())
                if strategy_action == 0:
                    regenerate_attempts += 1
                    offspring_x, offspring_pred, offspring_sigma, offspring_groups, offspring_info = generate_offspring_pool(
                        args=args,
                        nsga_problem=nsga2_problem,
                        archive_x=archive_x,
                        archive_y=archive_y,
                        step=step + regenerate_attempts,
                    )
                    continue
                selected_idx, _ = db_saea_agent.select_candidate_from_action(
                    action_idx=int(strategy_action),
                    archive_y=archive_y,
                    candidate_mean=offspring_pred,
                    candidate_std=offspring_sigma,
                    seed=int(args.seed) + step + regenerate_attempts,
                )
                if selected_idx is None:
                    raise RuntimeError(f"DB-SAEA strategy action {strategy_action} did not produce a candidate index.")
                break
        elif policy_name.lower() == "boformer":
            selected_idx = 0
        elif policy_name.lower() == "moead_ego":
            selected_idx = 0
        elif infill_criterion is not None:
            selected_idx, _ = _call_infill_select_index(
                infill_criterion,
                **_make_infill_selection_kwargs(
                    criterion=infill_criterion,
                    archive_y=archive_y,
                    candidate_mean=offspring_pred,
                    candidate_std=offspring_sigma,
                    seed=int(args.seed) + step,
                    offspring_info=offspring_info,
                    progress_ratio=fe_progress(
                        init_fe=int(args.init_fe),
                        step=int(step),
                        max_fe=int(args.max_fe),
                    ),
                ),
            )
        else:
            raise ValueError(f"Unsupported policy_name: {policy_name}")

        previous_front = pareto_front(np.asarray(archive_y, dtype=np.float32))
        selected_x = offspring_x[selected_idx : selected_idx + 1]
        selected_pred = offspring_pred[selected_idx]
        selected_sigma = offspring_sigma[selected_idx]
        selected_true = np.asarray(problem.evaluate(selected_x), dtype=np.float32)
        step_reward = compute_test_reward(
            reward_scheme_id=int(reward_scheme_id),
            previous_front=previous_front,
            selected_objectives=selected_true,
            ref_point=ref_point,
            reward_lambda=float(args.reward_lambda),
            true_pareto_hv=true_pareto_hv,
        )

        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
        archive_y_history, archive_sigma_history = append_archive_prediction_history(
            archive_y_history,
            archive_sigma_history,
            selected_pred=selected_pred,
            selected_sigma=selected_sigma,
            selected_true=selected_true,
            step=step,
        )
        hv = hypervolume(archive_y, ref_point)
        fe = int(args.init_fe) + step + 1
        front_size = int(pareto_front(archive_y).shape[0])
        fe_history.append(fe)
        hv_history.append(float(hv))

        record = StepRecord(
            step=step + 1,
            fe=fe,
            selected_index=selected_idx,
            selected_x=selected_x.reshape(-1).astype(float).tolist(),
            surrogate_y=selected_pred.astype(float).tolist(),
            true_y=selected_true.reshape(-1).astype(float).tolist(),
            reward=step_reward,
            hv=float(hv),
            archive_size=int(archive_x.shape[0]),
        )
        history.append(record)
        step_rewards.append(step_reward)
        if policy_key == "boformer":
            target_n_objectives = int(getattr(disc, "n_objectives", selected_pred.shape[0]))
            if boformer_batch_cache is None or int(boformer_batch_cache["x"].shape[0]) <= 0:
                raise RuntimeError("BOFormer rollout expected a cached selection.")
            history_mean_entry, history_variance_entry, _ = _build_boformer_history_entry(
                archive_y=previous_front,
                selected_mean=selected_pred,
                selected_variance_source=selected_sigma,
                target_n_objectives=target_n_objectives,
            )
            boformer_history_mean.append(history_mean_entry)
            boformer_history_variance.append(history_variance_entry)
            boformer_history_best.append(np.asarray(boformer_batch_cache["best"][0], dtype=np.float32))
            boformer_history_progress.append(float(boformer_batch_cache["progress"][0]))
            boformer_history_rewards.append(float(step_reward))
            boformer_history_q_values.append(float(boformer_batch_cache["q_values"][0]))
            boformer_history_objective_mask.append(np.asarray(boformer_batch_cache["objective_mask"][0], dtype=np.float32))

        log_line = f"{prefix}iter {record.step} | front = {front_size} | HV = {record.hv:.12f}"
        log_line += f" | reward = {record.reward:.6f}"
        logger(log_line)
        if policy_name.lower() == "boformer" and boformer_batch_cache is not None:
            boformer_batch_cache["cursor"] = int(boformer_batch_cache["cursor"]) + 1
            if int(boformer_batch_cache["cursor"]) >= int(boformer_batch_cache["x"].shape[0]):
                boformer_batch_cache = None
    final_front = pareto_front(archive_y)
    plot_path = None
    npy_paths = None
    if make_plot:
        plot_path = plot_results(
            args=args,
            fe_history=fe_history,
            hv_history=hv_history,
            archive_y=archive_y,
            true_pareto=true_pareto,
        )
        npy_paths = save_npy_outputs(
            args=args,
            archive_x=archive_x,
            archive_y=archive_y,
            final_front=final_front,
            fe_history=fe_history,
            hv_history=hv_history,
        )

    summary = {
        "problem": args.problem,
        "dim": int(args.dim),
        "seed": int(args.seed),
        "max_fe": int(args.max_fe),
        "init_fe": int(args.init_fe),
        "evolution_fe": n_evo_steps,
        "batch": int(getattr(args, "batch", 1)),
        "surrogate_model": surrogate_model_name(args),
        "candidate_solver": str(getattr(args, "solver", "nsga2")).lower(),
        "reward_lambda": float(args.reward_lambda),
        "reward_scheme": int(reward_scheme_id),
        "agent_name": args.agent_name,
        "policy_name": policy_name.lower(),
        "agent_pth": args.agent_pth,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": float(hypervolume(archive_y, ref_point)),
        "mean_reward_40_steps": float(np.mean(step_rewards)) if len(step_rewards) > 0 else 0.0,
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "plot_path": plot_path,
        "npy_paths": npy_paths,
        "history": [asdict(item) for item in history],
        "fe_history": fe_history,
        "hv_history": hv_history,
    }
    return summary, archive_y


def build_initial_archive(
    *,
    args: argparse.Namespace,
    problem,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(args.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    return np.asarray(archive_x, dtype=np.float32), np.asarray(archive_y, dtype=np.float32)


def run_policy_rollout_once(
    *,
    args: argparse.Namespace,
    problem,
    nsga2_problem,
    ref_point: np.ndarray,
    true_pareto: np.ndarray | None,
    policy_name: str,
    disc: Any | None = None,
    db_saea_agent: Any | None = None,
    infill_criterion: Any | None = None,
    compare_mode: bool = False,
    logger=print,
    run_label: str | None = None,
    reward_scheme_id: int = 1,
    true_pareto_hv: float | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    del run_label
    run_started_at = time.perf_counter()
    archive_x_init, archive_y_init = build_initial_archive(
        args=args,
        problem=problem,
        seed=int(args.seed),
    )
    summary, archive_y = run_policy_rollout(
        args=args,
        problem=problem,
        nsga2_problem=nsga2_problem,
        ref_point=ref_point,
        true_pareto=true_pareto,
        archive_x_init=archive_x_init,
        archive_y_init=archive_y_init,
        policy_name=policy_name,
        disc=disc,
        db_saea_agent=db_saea_agent,
        infill_criterion=infill_criterion,
        compare_mode=compare_mode,
        make_plot=False,
        logger=logger,
        reward_scheme_id=int(reward_scheme_id),
        true_pareto_hv=true_pareto_hv,
    )
    summary = dict(summary)
    summary["wall_clock_sec"] = float(time.perf_counter() - run_started_at)
    return summary, np.asarray(archive_y, dtype=np.float32)


def load_true_pareto_front(
    problem_name: str,
    dim: int,
    n_obj: int,
    n_points: int = 400,
) -> np.ndarray | None:
    try:
        problem = make_problem(problem_name, dim=int(dim))
        if hasattr(problem, "pareto_front"):
            pareto = problem.pareto_front(n_pareto_points=int(n_points))
            if pareto is not None:
                pareto = np.asarray(pareto, dtype=np.float32)
                if pareto.ndim == 2 and pareto.shape[1] >= int(n_obj):
                    return pareto[:, : int(n_obj)]
    except Exception:
        pass

    try:
        from pymoo.problems import get_problem
    except Exception:
        return None

    key = str(problem_name).lower()
    try:
        pymoo_problem = get_problem(key, n_var=int(dim), n_obj=int(n_obj))
    except TypeError:
        try:
            pymoo_problem = get_problem(key, n_var=int(dim))
        except Exception:
            return None
    except Exception:
        return None

    try:
        pareto = pymoo_problem.pareto_front(n_pareto_points=int(n_points))
    except TypeError:
        try:
            pareto = pymoo_problem.pareto_front()
        except Exception:
            return None
    except Exception:
        return None

    if pareto is None:
        return None
    pareto = np.asarray(pareto, dtype=np.float32)
    if pareto.ndim != 2 or pareto.shape[1] < int(n_obj):
        return None
    return pareto[:, : int(n_obj)]


def build_plot_command_caption() -> str:
    command_text = "python " + " ".join(str(part) for part in sys.argv[1:])
    wrapped = textwrap.wrap(command_text, width=120)
    return "\n".join(wrapped)


def attach_plot_caption(fig) -> None:
    caption = build_plot_command_caption()
    fig.text(
        0.01,
        0.01,
        caption,
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )


def show_plot(fig) -> None:
    try:
        width_in, height_in = fig.get_size_inches()
        dpi = fig.get_dpi()
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "resize"):
            manager.resize(int(width_in * dpi), int(height_in * dpi))
    except Exception:
        pass
    plt.show()


def resolve_pdf_plot_path(plot_file: Path, args: argparse.Namespace) -> Path:
    return default_test_plot_dir(args, kind="pdf") / f"{plot_file.stem}.pdf"


def save_plot_outputs(
    fig,
    *,
    plot_file: Path,
    args: argparse.Namespace,
    with_command_caption_on_png: bool,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> Path:
    pdf_file = resolve_pdf_plot_path(plot_file, args)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_layout_rect)
    fig.savefig(pdf_file)
    if with_command_caption_on_png:
        attach_plot_caption(fig)
        # Do not reserve extra bottom space here: it changes the effective plot area
        # away from the requested 4:3 figure ratio.
    fig.savefig(plot_file, dpi=180)
    return pdf_file.resolve()


PLOT_WIDTH_4_3 = 8.0
PLOT_HEIGHT_4_3 = 6.0
WIDE_PLOT_WIDTH_4_3 = 16.0
WIDE_PLOT_HEIGHT_4_3 = 6.0
TRUE_PARETO_GLASS_COLOR = "#9fd3f2"


def _normalize_plot_label(label: str) -> str:
    return str(label).strip().lower().replace("_", "-")


def _show_hv_popup(
    *,
    fe_histories: list[list[int]],
    hv_histories: list[list[float]],
    labels: list[str],
    colors: list[str],
    markers: list[str] | None = None,
    hv_std_histories: list[list[float]] | None = None,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(PLOT_WIDTH_4_3, PLOT_HEIGHT_4_3))
    marker_list = markers if markers is not None else ["o"] * max(1, len(labels))
    for idx, label in enumerate(labels):
        color = colors[idx % len(colors)]
        marker = marker_list[idx % len(marker_list)]
        if hv_std_histories is not None and idx < len(hv_std_histories) and hv_std_histories[idx] is not None:
            fe_arr = np.asarray(fe_histories[idx], dtype=np.float64)
            hv_arr = np.asarray(hv_histories[idx], dtype=np.float64)
            std_arr = np.asarray(hv_std_histories[idx], dtype=np.float64)
            if fe_arr.shape == hv_arr.shape == std_arr.shape:
                ax.fill_between(
                    fe_arr,
                    hv_arr - std_arr,
                    hv_arr + std_arr,
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                )
        ax.plot(
            fe_histories[idx],
            hv_histories[idx],
            marker=marker,
            linewidth=1.8,
            markersize=4,
            color=color,
            label=label,
        )
    ax.set_xlabel("FE")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(labels) > 1:
        ax.legend()
    show_plot(fig)
    plt.close(fig)


def _show_pareto_popup(
    *,
    archive_ys: list[np.ndarray],
    fronts: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    markers: list[str] | None = None,
    true_pareto: np.ndarray | None,
    title: str,
) -> None:
    n_obj = int(np.asarray(archive_ys[0], dtype=np.float32).shape[1])
    fig = plt.figure(figsize=(PLOT_WIDTH_4_3, PLOT_HEIGHT_4_3))
    if n_obj == 3:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        ax = fig.add_subplot(1, 1, 1)
    marker_list = markers if markers is not None else ["o"] * max(1, len(labels))

    if n_obj == 2:
        for idx, label in enumerate(labels):
            color = colors[idx % len(colors)]
            marker = marker_list[idx % len(marker_list)]
            archive_y = np.asarray(archive_ys[idx], dtype=np.float32)
            front = np.asarray(fronts[idx], dtype=np.float32)
            ax.scatter(archive_y[:, 0], archive_y[:, 1], s=12, alpha=0.14, color=color, label=f"{label} Archive")
            ax.scatter(front[:, 0], front[:, 1], s=28, alpha=0.95, marker=marker, color=color, label=f"{label} PF")
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax.plot(
                true_pareto[order, 0],
                true_pareto[order, 1],
                linewidth=2.0,
                color=TRUE_PARETO_GLASS_COLOR,
                alpha=0.65,
                label="True PF",
            )
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.grid(True, alpha=0.3)
    elif n_obj == 3:
        for idx, label in enumerate(labels):
            color = colors[idx % len(colors)]
            marker = marker_list[idx % len(marker_list)]
            archive_y = np.asarray(archive_ys[idx], dtype=np.float32)
            front = np.asarray(fronts[idx], dtype=np.float32)
            ax.scatter(archive_y[:, 0], archive_y[:, 1], archive_y[:, 2], s=10, alpha=0.12, color=color, label=f"{label} Archive")
            ax.scatter(front[:, 0], front[:, 1], front[:, 2], s=26, alpha=0.95, marker=marker, color=color, label=f"{label} PF")
        if true_pareto is not None and true_pareto.shape[1] >= 3:
            ax.scatter(
                true_pareto[:, 0],
                true_pareto[:, 1],
                true_pareto[:, 2],
                s=8,
                alpha=0.28,
                color=TRUE_PARETO_GLASS_COLOR,
                label="True PF",
            )
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
    else:
        raise ValueError(f"_show_pareto_popup currently supports only 2 or 3 objectives, got n_obj={n_obj}.")

    ax.set_title(title)
    ax.legend()
    show_plot(fig)
    plt.close(fig)


def _resolve_series_colors(labels: list[str], plot_tag: str | None = None) -> list[str]:
    _ = plot_tag
    explicit_colors = {
        "disc": "#1f77b4",
        "disc-af": "#e76f51",
        "disc w/o candidate pool interaction": "#e76f51",
        "disc-af2": "#e377c2",
        "db-saea": "#f2b300",
        "boformer": "#d62728",
        "nsga2": "#7f7f7f",
        "nsga-ii": "#7f7f7f",
        "moea-d/ego": "#8c564b",
        "moea/d-ego": "#8c564b",
        "moead-ego": "#8c564b",
        "usemo-uncertainty": "#F26D5B",
    }
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    cycle_colors = [] if prop_cycle is None else list(prop_cycle.by_key().get("color", []))
    if len(cycle_colors) <= 0:
        cycle_colors = [str(color) for color in plt.cm.tab10.colors]
    resolved: list[str] = []
    fallback_idx = 0
    for label in labels:
        key = str(label).strip().lower().replace("_", "-")
        color = explicit_colors.get(key)
        if color is None:
            color = str(cycle_colors[fallback_idx % len(cycle_colors)])
            fallback_idx += 1
        resolved.append(str(color))
    return resolved


def plot_results(
    *,
    args: argparse.Namespace,
    fe_history: list[int],
    hv_history: list[float],
    archive_y: np.ndarray,
    true_pareto: np.ndarray | None,
) -> str:
    agent_tag = resolve_primary_output_tag(args)
    plot_path = args.plot_path
    if plot_path is None:
        plot_dir = default_test_plot_dir(args, kind="png")
        plot_path = str(plot_dir / f"test_{agent_tag}_{args.problem.lower()}_seed{int(args.seed)}.png")
    else:
        plot_path = str(Path(plot_path))

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    archive_front = pareto_front(archive_y)
    n_obj = int(archive_y.shape[1])

    fig = plt.figure(figsize=(WIDE_PLOT_WIDTH_4_3, WIDE_PLOT_HEIGHT_4_3))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4)
    ax_hv.set_xlabel("FE")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title(f"{args.problem} HV vs FE")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], s=18, alpha=0.45, label="Archive")
        ax_pf.scatter(archive_front[:, 0], archive_front[:, 1], s=26, alpha=0.9, label="Archive PF")
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax_pf.plot(
                true_pareto[order, 0],
                true_pareto[order, 1],
                linewidth=2.0,
                color=TRUE_PARETO_GLASS_COLOR,
                alpha=0.65,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.grid(True, alpha=0.3)
    elif n_obj == 3:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], archive_y[:, 2], s=18, alpha=0.30, label="Archive")
        ax_pf.scatter(
            archive_front[:, 0],
            archive_front[:, 1],
            archive_front[:, 2],
            s=28,
            alpha=0.95,
            label="Archive PF",
        )
        if true_pareto is not None and true_pareto.shape[1] >= 3:
            ax_pf.scatter(
                true_pareto[:, 0],
                true_pareto[:, 1],
                true_pareto[:, 2],
                s=8,
                alpha=0.28,
                color=TRUE_PARETO_GLASS_COLOR,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    else:
        raise ValueError(f"plot_results currently supports only 2 or 3 objectives, got n_obj={n_obj}.")

    ax_pf.set_title(f"{args.problem} Archive vs True PF")
    ax_pf.legend()

    save_plot_outputs(
        fig,
        plot_file=plot_file,
        args=args,
        with_command_caption_on_png=False,
    )
    plt.close(fig)
    _show_hv_popup(
        fe_histories=[list(fe_history)],
        hv_histories=[list(hv_history)],
        labels=["HV"],
        colors=["#1f77b4"],
        title=f"{args.problem} HV vs FE",
        ylabel="Hypervolume",
    )
    _show_pareto_popup(
        archive_ys=[np.asarray(archive_y, dtype=np.float32)],
        fronts=[np.asarray(archive_front, dtype=np.float32)],
        labels=["Archive"],
        colors=["#1f77b4"],
        true_pareto=true_pareto,
        title=f"{args.problem} Archive vs True PF",
    )
    return str(plot_file.resolve())


def plot_compare_results(
    *,
    args: argparse.Namespace,
    primary_fe_history: list[int],
    primary_hv_history: list[float],
    primary_archive_y: np.ndarray,
    baseline_fe_history: list[int],
    baseline_hv_history: list[float],
    baseline_archive_y: np.ndarray,
    primary_label: str,
    baseline_label: str,
    true_pareto: np.ndarray | None,
    plot_tag: str | None = None,
) -> str:
    return plot_multi_compare_results(
        args=args,
        labels=[primary_label, baseline_label],
        fe_histories=[primary_fe_history, baseline_fe_history],
        hv_histories=[primary_hv_history, baseline_hv_history],
        archive_ys=[primary_archive_y, baseline_archive_y],
        true_pareto=true_pareto,
        plot_tag=plot_tag,
    )


def plot_multi_compare_results(
    *,
    args: argparse.Namespace,
    labels: list[str],
    fe_histories: list[list[int]],
    hv_histories: list[list[float]],
    hv_std_histories: list[list[float]] | None = None,
    archive_ys: list[np.ndarray],
    true_pareto: np.ndarray | None,
    plot_tag: str | None = None,
) -> str:
    agent_tag = resolve_primary_output_tag(args)
    plot_path = args.plot_path
    if plot_path is None:
        suffix = "" if plot_tag is None else f"_{str(plot_tag).strip().lower().replace('-', '_')}"
        plot_dir = default_test_plot_dir(args, kind="png")
        plot_path = str(plot_dir / f"test_{agent_tag}_{args.problem.lower()}_seed{int(args.seed)}_compare{suffix}.png")
    else:
        plot_file = Path(plot_path)
        suffix = "" if plot_tag is None else f"_{str(plot_tag).strip().lower().replace('-', '_')}"
        plot_file = plot_file.with_name(f"{plot_file.stem}_compare{suffix}{plot_file.suffix}")
        plot_path = str(plot_file)

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    if not (len(labels) == len(fe_histories) == len(hv_histories) == len(archive_ys)):
        raise ValueError("labels, fe_histories, hv_histories, and archive_ys must have the same length.")
    if len(labels) <= 0:
        raise ValueError("plot_multi_compare_results requires at least one result series.")

    fronts = [pareto_front(np.asarray(archive_y, dtype=np.float32)) for archive_y in archive_ys]
    n_obj = int(np.asarray(archive_ys[0], dtype=np.float32).shape[1])
    colors = _resolve_series_colors(labels, plot_tag=plot_tag)
    markers = ["o", "s", "^", "D", "x", "P", "v", "<", ">", "*", "h", "X"]

    fig = plt.figure(figsize=(WIDE_PLOT_WIDTH_4_3, WIDE_PLOT_HEIGHT_4_3))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    for idx, label in enumerate(labels):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        if hv_std_histories is not None and idx < len(hv_std_histories) and hv_std_histories[idx] is not None:
            fe_arr = np.asarray(fe_histories[idx], dtype=np.float64)
            hv_arr = np.asarray(hv_histories[idx], dtype=np.float64)
            std_arr = np.asarray(hv_std_histories[idx], dtype=np.float64)
            if fe_arr.shape == hv_arr.shape == std_arr.shape:
                ax_hv.fill_between(
                    fe_arr,
                    hv_arr - std_arr,
                    hv_arr + std_arr,
                    color=color,
                    alpha=0.10,
                    linewidth=0.0,
                )
        ax_hv.plot(
            fe_histories[idx],
            hv_histories[idx],
            marker=marker,
            linewidth=1.8,
            markersize=4,
            color=color,
            label=label,
        )
    ax_hv.set_xlabel("FE")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title(f"{args.problem} HV Comparison")
    ax_hv.grid(True, alpha=0.3)
    ax_hv.legend()

    if n_obj == 2:
        for idx, label in enumerate(labels):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            archive_y = np.asarray(archive_ys[idx], dtype=np.float32)
            front = fronts[idx]
            ax_pf.scatter(
                archive_y[:, 0],
                archive_y[:, 1],
                s=12,
                alpha=0.14,
                color=color,
                label=f"{label} Archive",
            )
            ax_pf.scatter(
                front[:, 0],
                front[:, 1],
                s=28,
                alpha=0.95,
                marker=marker,
                color=color,
                label=f"{label} PF",
            )
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax_pf.plot(
                true_pareto[order, 0],
                true_pareto[order, 1],
                linewidth=2.0,
                color=TRUE_PARETO_GLASS_COLOR,
                alpha=0.65,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.grid(True, alpha=0.3)
    elif n_obj == 3:
        for idx, label in enumerate(labels):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            archive_y = np.asarray(archive_ys[idx], dtype=np.float32)
            front = fronts[idx]
            ax_pf.scatter(
                archive_y[:, 0],
                archive_y[:, 1],
                archive_y[:, 2],
                s=10,
                alpha=0.12,
                color=color,
                label=f"{label} Archive",
            )
            ax_pf.scatter(
                front[:, 0],
                front[:, 1],
                front[:, 2],
                s=26,
                alpha=0.95,
                marker=marker,
                color=color,
                label=f"{label} PF",
            )
        if true_pareto is not None and true_pareto.shape[1] >= 3:
            ax_pf.scatter(
                true_pareto[:, 0],
                true_pareto[:, 1],
                true_pareto[:, 2],
                s=8,
                alpha=0.28,
                color=TRUE_PARETO_GLASS_COLOR,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    else:
        raise ValueError(f"plot_compare_results currently supports only 2 or 3 objectives, got n_obj={n_obj}.")

    ax_pf.set_title(f"{args.problem} Archive Comparison")
    ax_pf.legend()
    save_plot_outputs(
        fig,
        plot_file=plot_file,
        args=args,
        with_command_caption_on_png=True,
    )
    plt.close(fig)
    _show_hv_popup(
        fe_histories=[list(history) for history in fe_histories],
        hv_histories=[list(history) for history in hv_histories],
        labels=list(labels),
        colors=list(colors),
        markers=list(markers),
        hv_std_histories=None if hv_std_histories is None else [list(history) if history is not None else [] for history in hv_std_histories],
        title=f"{args.problem} HV Comparison",
        ylabel="Hypervolume",
    )
    _show_pareto_popup(
        archive_ys=[np.asarray(archive_y, dtype=np.float32) for archive_y in archive_ys],
        fronts=[np.asarray(front, dtype=np.float32) for front in fronts],
        labels=list(labels),
        colors=list(colors),
        markers=list(markers),
        true_pareto=true_pareto,
        title=f"{args.problem} Archive Comparison",
    )
    return str(plot_file.resolve())


def save_npy_outputs(
    *,
    args: argparse.Namespace,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    final_front: np.ndarray,
    fe_history: list[int],
    hv_history: list[float],
) -> dict[str, str]:
    out_dir = Path("npy")
    out_dir.mkdir(parents=True, exist_ok=True)
    agent_tag = resolve_primary_output_tag(args)
    stem = f"test_{agent_tag}_{args.problem.lower()}_seed{int(args.seed)}"

    paths = {
        "archive_x": out_dir / f"{stem}_archive_x.npy",
        "archive_y": out_dir / f"{stem}_archive_y.npy",
        "final_front": out_dir / f"{stem}_final_front.npy",
        "fe_history": out_dir / f"{stem}_fe_history.npy",
        "hv_history": out_dir / f"{stem}_hv_history.npy",
    }

    np.save(paths["archive_x"], np.asarray(archive_x, dtype=np.float32))
    np.save(paths["archive_y"], np.asarray(archive_y, dtype=np.float32))
    np.save(paths["final_front"], np.asarray(final_front, dtype=np.float32))
    np.save(paths["fe_history"], np.asarray(fe_history, dtype=np.int64))
    np.save(paths["hv_history"], np.asarray(hv_history, dtype=np.float64))

    return {key: str(path.resolve()) for key, path in paths.items()}


def main(agent_name: str = "disc") -> None:
    args = parse_args()
    args.agent_name = str(agent_name).lower()
    primary_policy_name = resolve_primary_policy_name(args)
    if (
        is_learned_infill_policy(primary_policy_name)
        and not bool(args.random_model)
        and not getattr(args, "agent_pth", None)
    ):
        args.agent_pth = resolve_default_best_reward_checkpoint(args)
    args.seed_input = str(args.seed)
    args.seed = resolve_seed(args.seed)
    set_seed(int(args.seed))
    test_log_path = default_test_log_path(args, agent_name=args.agent_name)
    log, log_fp = make_test_logger(test_log_path)

    try:
        if test_log_path is not None:
            log(f"test_log_path = {str(test_log_path.resolve())}")
        log(
            f"test | problem = {args.problem} | dim = {int(args.dim)} | "
            f"seed = {int(args.seed)} | solver = {str(args.solver).lower()}"
        )
        comparison_family, run_variant = resolve_solver_baseline_metadata(args)
        solver_family = canonical_solver_family_name(getattr(args, "solver", None))
        problem = make_problem(args.problem, dim=int(args.dim))
        n_evo_steps = int(args.max_fe) - int(args.init_fe)
        reward_scheme_id = resolve_test_reward_scheme(args)
        if args.reward_lambda is None:
            args.reward_lambda = resolve_default_reward_lambda(int(reward_scheme_id))

        archive_x, archive_y = build_initial_archive(
            args=args,
            problem=problem,
            seed=int(args.seed),
        )
        n_obj = int(archive_y.shape[1])
        ref_point = get_reference_point(args.problem, n_obj=n_obj)
        nsga2_problem = make_nsga2_problem_adapter(problem, n_obj)
        true_pareto = load_true_pareto_front(args.problem, int(args.dim), n_obj)
        true_pareto_hv = None
        if int(reward_scheme_id) == 2:
            true_pareto_hv = get_true_pareto_hv(args.problem, dim=int(args.dim), n_obj=n_obj)
        primary_policy_name = resolve_primary_policy_name(args)
        primary_label = primary_policy_display_name(primary_policy_name)
        primary_inference_source = resolve_inference_source(
            policy_name=primary_policy_name,
            agent_pth=getattr(args, "agent_pth", None),
            random_model=bool(args.random_model),
        )
        primary_infill = (
            None
            if is_learned_infill_policy(primary_policy_name)
            else build_compare_infill_criterion(primary_policy_name, ref_point=ref_point)
        )
        compare_infill_names = resolve_compare_infill_names(args)
        compare_infill_name = resolve_compare_infill_name(args)
        compare_algo_names = resolve_compare_algo_names(args)
        compare_algo_name = resolve_compare_algo_name(args)
        compare_random_model = bool(getattr(args, "compare_random_model", False)) or (
            bool(args.random_model) and getattr(args, "compare_agent_pth", None) is None
        )
        compare_inference_source = None
        if len(compare_infill_names) == 1:
            compare_inference_source = resolve_inference_source(
                policy_name=str(compare_infill_name),
                agent_pth=(getattr(args, "compare_agent_pth", None) if is_db_saea_policy(compare_infill_name) else None),
                random_model=bool(compare_random_model) if is_db_saea_policy(compare_infill_name) else False,
            )
        elif len(compare_algo_names) == 1:
            compare_inference_source = resolve_inference_source(
                policy_name="disc",
                agent_pth=getattr(args, "compare_agent_pth", None),
                random_model=bool(compare_random_model),
                compare_algo=str(compare_algo_name),
            )
        if int(reward_scheme_id) == 2 and true_pareto_hv is None:
            raise RuntimeError(
                f"No precomputed true Pareto HV found for {args.problem}-{int(args.dim)}D-{int(n_obj)}obj in problem/problem.py."
            )
        disc = None
        agent_load_breakdown = {
            "model_to_device_sec": 0.0,
            "torch_load_sec": 0.0,
            "load_state_dict_sec": 0.0,
        }
        if primary_policy_name in {"disc_af", "disc_af2", "disc_single_dqn", "boformer", "db_saea"}:
            args.agent_name = primary_policy_name
        db_saea_agent = None
        if is_db_saea_policy(primary_policy_name):
            db_saea_agent, agent_load_breakdown = build_disc(
                args,
                map_location=str(args.device),
                agent_name="db_saea",
                n_objectives=int(n_obj),
            )
        elif is_agent_policy(primary_policy_name):
            disc, agent_load_breakdown = build_disc(
                args,
                map_location=str(args.device),
                agent_name=args.agent_name,
                n_objectives=int(n_obj),
            )
        primary_summary, primary_archive_y = run_policy_rollout_once(
            args=args,
            problem=problem,
            nsga2_problem=nsga2_problem,
            ref_point=ref_point,
            true_pareto=true_pareto,
            policy_name=primary_policy_name,
            disc=disc,
            db_saea_agent=db_saea_agent,
            infill_criterion=primary_infill,
            compare_mode=bool(compare_infill_names or compare_algo_names),
            logger=log,
            run_label=primary_label,
            reward_scheme_id=int(reward_scheme_id),
            true_pareto_hv=true_pareto_hv,
        )
        if len(compare_infill_names) > 0 or len(compare_algo_names) > 0:
            infill_results: dict[str, dict[str, Any]] = {}
            infill_archives: dict[str, np.ndarray] = {}
            for infill_name in compare_infill_names:
                infill_agent = None
                infill_criterion_obj = None
                if is_db_saea_policy(infill_name):
                    compare_args = argparse.Namespace(**vars(args))
                    compare_args.agent_pth = getattr(args, "compare_agent_pth", None)
                    compare_args.random_model = bool(compare_random_model)
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError("compare_infill=db_saea requires --compare_agent_pth unless --compare_random_model is used.")
                    infill_agent, _ = build_disc(
                        compare_args,
                        map_location=str(args.device),
                        agent_name="db_saea",
                        n_objectives=int(n_obj),
                    )
                else:
                    infill_criterion_obj = build_compare_infill_criterion(infill_name, ref_point=ref_point)
                infill_summary, infill_archive_y = run_policy_rollout_once(
                    args=args,
                    problem=problem,
                    nsga2_problem=nsga2_problem,
                    ref_point=ref_point,
                    true_pareto=true_pareto,
                    policy_name=str(infill_name),
                    db_saea_agent=infill_agent if is_db_saea_policy(infill_name) else None,
                    infill_criterion=infill_criterion_obj,
                    compare_mode=True,
                    logger=log,
                    run_label=compare_infill_display_name(infill_name),
                    reward_scheme_id=int(reward_scheme_id),
                    true_pareto_hv=true_pareto_hv,
                )
                infill_results[str(infill_name)] = infill_summary
                infill_archives[str(infill_name)] = infill_archive_y
            baseline_results: dict[str, dict[str, Any]] = {}
            baseline_archives: dict[str, np.ndarray] = {}
            for algo_name in compare_algo_names:
                compare_args = argparse.Namespace(**vars(args))
                compare_args.agent_pth = args.compare_agent_pth
                compare_args.random_model = bool(compare_random_model)
                baseline_agent = None
                baseline_policy_name = str(algo_name)
                baseline_infill = None
                if algo_name == "disc":
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError("compare_algo=disc requires --compare_agent_pth unless --random_model is used.")
                    compare_args.agent_name = str(args.agent_name)
                    baseline_agent, _ = build_disc(compare_args, map_location=str(args.device), agent_name=str(args.agent_name))
                elif algo_name in {"disc_af", "disc_af2", "disc_single_dqn"}:
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError(
                            f"compare_algo={algo_name} requires --compare_agent_pth unless --random_model is used."
                        )
                    compare_args.agent_name = str(algo_name)
                    baseline_agent, _ = build_disc(
                        compare_args,
                        map_location=str(args.device),
                        agent_name=str(algo_name),
                    )
                elif algo_name == "db_saea":
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError("compare_algo=db_saea requires --compare_agent_pth unless --random_model is used.")
                    baseline_agent, _ = build_disc(compare_args, map_location=str(args.device), agent_name="db_saea")
                elif algo_name == "moead_ego":
                    compare_args.solver = "nsga3"
                    compare_args.surrogate_model = "gp"
                else:
                    raise ValueError(f"Unsupported compare_algo: {algo_name}")
                baseline_summary, baseline_archive_y = run_policy_rollout_once(
                    args=compare_args,
                    problem=problem,
                    nsga2_problem=nsga2_problem,
                    ref_point=ref_point,
                    true_pareto=true_pareto,
                    policy_name=baseline_policy_name,
                    disc=baseline_agent if algo_name in {"disc", "disc_af", "disc_af2", "disc_single_dqn"} else None,
                    db_saea_agent=baseline_agent if algo_name == "db_saea" else None,
                    infill_criterion=baseline_infill,
                    compare_mode=True,
                    logger=log,
                    run_label=compare_algo_display_name(algo_name),
                    reward_scheme_id=int(reward_scheme_id),
                    true_pareto_hv=true_pareto_hv,
                )
                baseline_results[str(algo_name)] = baseline_summary
                baseline_archives[str(algo_name)] = baseline_archive_y
            primary_compare_label = primary_label
            compare_plot_path = None
            compare_plot_paths: dict[str, str] = {}
            if len(compare_infill_names) > 0:
                infill_plot_labels = [primary_compare_label] + [
                    compare_infill_display_name(infill_name) for infill_name in compare_infill_names
                ]
                infill_plot_fe_histories = [list(primary_summary["fe_history"])] + [
                    list(infill_results[infill_name]["fe_history"]) for infill_name in compare_infill_names
                ]
                infill_plot_hv_histories = [list(primary_summary["hv_history"])] + [
                    list(infill_results[infill_name]["hv_history"]) for infill_name in compare_infill_names
                ]
                infill_plot_hv_std_histories = [list(primary_summary.get("hv_std_history", []))] + [
                    list(infill_results[infill_name].get("hv_std_history", [])) for infill_name in compare_infill_names
                ]
                infill_plot_archives = [primary_archive_y] + [
                    infill_archives[infill_name] for infill_name in compare_infill_names
                ]
                compare_plot_paths["infill"] = plot_multi_compare_results(
                    args=args,
                    labels=infill_plot_labels,
                    fe_histories=infill_plot_fe_histories,
                    hv_histories=infill_plot_hv_histories,
                    hv_std_histories=infill_plot_hv_std_histories,
                    archive_ys=infill_plot_archives,
                    true_pareto=true_pareto,
                    plot_tag="infill",
                )
            if len(compare_algo_names) > 0:
                algo_plot_labels = [primary_compare_label] + [
                    compare_algo_display_name(algo_name) for algo_name in compare_algo_names
                ]
                algo_plot_fe_histories = [list(primary_summary["fe_history"])] + [
                    list(baseline_results[algo_name]["fe_history"]) for algo_name in compare_algo_names
                ]
                algo_plot_hv_histories = [list(primary_summary["hv_history"])] + [
                    list(baseline_results[algo_name]["hv_history"]) for algo_name in compare_algo_names
                ]
                algo_plot_hv_std_histories = [list(primary_summary.get("hv_std_history", []))] + [
                    list(baseline_results[algo_name].get("hv_std_history", [])) for algo_name in compare_algo_names
                ]
                algo_plot_archives = [primary_archive_y] + [
                    baseline_archives[algo_name] for algo_name in compare_algo_names
                ]
                compare_plot_paths["algo"] = plot_multi_compare_results(
                    args=args,
                    labels=algo_plot_labels,
                    fe_histories=algo_plot_fe_histories,
                    hv_histories=algo_plot_hv_histories,
                    hv_std_histories=algo_plot_hv_std_histories,
                    archive_ys=algo_plot_archives,
                    true_pareto=true_pareto,
                    plot_tag="algo",
                )
            compare_plot_path = compare_plot_paths.get("infill") or compare_plot_paths.get("algo")
            summary = {
                "problem": args.problem,
                "dim": int(args.dim),
                "seed_input": str(getattr(args, "seed_input", args.seed)),
                "seed": int(args.seed),
                "surrogate_model": surrogate_model_name(args),
                "agent_name": args.agent_name,
                "framework_label": resolve_framework_label(args),
                "infill_label": resolve_infill_label(args),
                "primary_policy": primary_policy_name,
                "primary_inference_source": primary_inference_source,
                "compare_infill": compare_infill_name,
                "compare_infill_resolved": compare_infill_names,
                "compare_algo": compare_algo_name,
                "compare_algo_resolved": compare_algo_names,
                "compare_inference_source": compare_inference_source,
                "reward_scheme": int(reward_scheme_id),
                "primary": primary_summary,
                "infill_results": infill_results,
                "baseline_results": baseline_results,
                "compare_plot_path": compare_plot_path,
                "compare_plot_paths": compare_plot_paths,
                "test_log_path": (str(test_log_path.resolve()) if test_log_path is not None else None),
                "comparison_family": comparison_family,
                "solver_family": solver_family,
                "run_variant": run_variant,
            }
            if len(compare_infill_names) == 1:
                summary["infill"] = infill_results[compare_infill_names[0]]
            if len(compare_algo_names) == 1:
                summary["baseline"] = baseline_results[compare_algo_names[0]]
            if is_learned_infill_policy(primary_policy_name):
                summary["disc"] = primary_summary
        elif len(compare_algo_names) > 0:
            baseline_results: dict[str, dict[str, Any]] = {}
            baseline_archives: dict[str, np.ndarray] = {}
            for algo_name in compare_algo_names:
                compare_args = argparse.Namespace(**vars(args))
                compare_args.agent_pth = args.compare_agent_pth
                compare_args.random_model = bool(compare_random_model)
                baseline_agent = None
                baseline_policy_name = str(algo_name)
                baseline_infill = None
                if algo_name == "disc":
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError("compare_algo=disc requires --compare_agent_pth unless --random_model is used.")
                    compare_args.agent_name = str(args.agent_name)
                    baseline_agent, _ = build_disc(compare_args, map_location=str(args.device), agent_name=str(args.agent_name))
                elif algo_name in {"disc_af", "disc_af2", "disc_single_dqn"}:
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError(
                            f"compare_algo={algo_name} requires --compare_agent_pth unless --random_model is used."
                        )
                    compare_args.agent_name = str(algo_name)
                    baseline_agent, _ = build_disc(
                        compare_args,
                        map_location=str(args.device),
                        agent_name=str(algo_name),
                    )
                elif algo_name == "db_saea":
                    if compare_args.agent_pth is None and not bool(compare_args.random_model):
                        raise ValueError("compare_algo=db_saea requires --compare_agent_pth unless --random_model is used.")
                    baseline_agent, _ = build_disc(compare_args, map_location=str(args.device), agent_name="db_saea")
                elif algo_name == "moead_ego":
                    compare_args.solver = "nsga3"
                    compare_args.surrogate_model = "gp"
                else:
                    raise ValueError(f"Unsupported compare_algo: {algo_name}")
                baseline_summary, baseline_archive_y = run_policy_rollout_once(
                    args=compare_args,
                    problem=problem,
                    nsga2_problem=nsga2_problem,
                    ref_point=ref_point,
                    true_pareto=true_pareto,
                    policy_name=baseline_policy_name,
                    disc=baseline_agent if algo_name in {"disc", "disc_af", "disc_af2", "disc_single_dqn"} else None,
                    db_saea_agent=baseline_agent if algo_name == "db_saea" else None,
                    infill_criterion=baseline_infill,
                    compare_mode=True,
                    logger=log,
                    run_label=compare_algo_display_name(algo_name),
                    reward_scheme_id=int(reward_scheme_id),
                    true_pareto_hv=true_pareto_hv,
                )
                baseline_results[str(algo_name)] = baseline_summary
                baseline_archives[str(algo_name)] = baseline_archive_y
            compare_plot_labels = [primary_label] + [compare_algo_display_name(algo_name) for algo_name in compare_algo_names]
            compare_plot_fe_histories = [list(primary_summary["fe_history"])] + [
                list(baseline_results[algo_name]["fe_history"]) for algo_name in compare_algo_names
            ]
            compare_plot_hv_histories = [list(primary_summary["hv_history"])] + [
                list(baseline_results[algo_name]["hv_history"]) for algo_name in compare_algo_names
            ]
            compare_plot_hv_std_histories = [list(primary_summary.get("hv_std_history", []))] + [
                list(baseline_results[algo_name].get("hv_std_history", [])) for algo_name in compare_algo_names
            ]
            compare_plot_archives = [primary_archive_y] + [
                baseline_archives[algo_name] for algo_name in compare_algo_names
            ]
            compare_plot_path = plot_multi_compare_results(
                args=args,
                labels=compare_plot_labels,
                fe_histories=compare_plot_fe_histories,
                hv_histories=compare_plot_hv_histories,
                hv_std_histories=compare_plot_hv_std_histories,
                archive_ys=compare_plot_archives,
                true_pareto=true_pareto,
            )
            summary = {
                "problem": args.problem,
                "dim": int(args.dim),
                "seed_input": str(getattr(args, "seed_input", args.seed)),
                "seed": int(args.seed),
                "surrogate_model": surrogate_model_name(args),
                "agent_name": args.agent_name,
                "framework_label": resolve_framework_label(args),
                "infill_label": resolve_infill_label(args),
                "primary_policy": primary_policy_name,
                "primary_inference_source": primary_inference_source,
                "compare_algo": compare_algo_name,
                "compare_algo_resolved": compare_algo_names,
                "compare_inference_source": compare_inference_source,
                "reward_scheme": int(reward_scheme_id),
                "primary": primary_summary,
                "baseline_results": baseline_results,
                "compare_plot_path": compare_plot_path,
                "test_log_path": (str(test_log_path.resolve()) if test_log_path is not None else None),
                "comparison_family": comparison_family,
                "solver_family": solver_family,
                "run_variant": run_variant,
            }
            if len(compare_algo_names) == 1:
                summary["baseline"] = baseline_results[compare_algo_names[0]]
            if is_learned_infill_policy(primary_policy_name):
                summary["disc"] = primary_summary
        else:
            summary = primary_summary
            summary["framework_label"] = resolve_framework_label(args)
            summary["infill_label"] = resolve_infill_label(args)
            summary["comparison_family"] = comparison_family
            summary["solver_family"] = solver_family
            summary["run_variant"] = run_variant
            summary["plot_path"] = plot_results(
                args=args,
                fe_history=list(primary_summary["fe_history"]),
                hv_history=list(primary_summary["hv_history"]),
                archive_y=primary_archive_y,
                true_pareto=true_pareto,
            )
            summary["npy_paths"] = None
            summary["seed_input"] = str(getattr(args, "seed_input", args.seed))
            summary["test_log_path"] = str(test_log_path.resolve()) if test_log_path is not None else None
            summary["primary_inference_source"] = primary_inference_source

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if len(compare_infill_names) > 0:
            reward_parts = [
                f"{primary_label} = {primary_summary['mean_reward_40_steps']:.6f}"
            ]
            hv_summary = {}
            algo_hv_summary = {}
            for infill_name in compare_infill_names:
                infill_result = summary["infill_results"][infill_name]
                reward_parts.append(
                    f"{compare_infill_display_name(infill_name)} = {infill_result['mean_reward_40_steps']:.6f}"
                )
                hv_summary[infill_name] = float(infill_result["final_hv"])
            for algo_name in compare_algo_names:
                baseline_result = summary["baseline_results"][algo_name]
                algo_label = compare_algo_display_name(algo_name)
                reward_parts.append(
                    f"{algo_label} = {baseline_result['mean_reward_40_steps']:.6f}"
                )
                algo_hv_summary[algo_name] = float(baseline_result["final_hv"])
            log(f"mean reward ({n_evo_steps} steps) | " + " | ".join(reward_parts))
            log(
                json.dumps(
                    {
                        "problem": summary["problem"],
                        "dim": summary["dim"],
                        "seed": summary["seed"],
                        "surrogate_model": summary["surrogate_model"],
                        "agent_name": summary["agent_name"],
                        "primary_policy": primary_policy_name,
                        "primary_inference_source": primary_inference_source,
                        "compare_infill": compare_infill_name,
                        "compare_infill_resolved": compare_infill_names,
                        "compare_algo": compare_algo_name,
                        "compare_algo_resolved": compare_algo_names,
                        "compare_inference_source": compare_inference_source,
                        "primary_final_hv": primary_summary["final_hv"],
                        "compare_final_hv": hv_summary,
                        "compare_algo_final_hv": algo_hv_summary,
                        "compare_plot_path": summary["compare_plot_path"],
                        "compare_plot_paths": summary.get("compare_plot_paths", {}),
                        "test_log_path": summary["test_log_path"],
                        "comparison_family": summary.get("comparison_family"),
                        "solver_family": summary.get("solver_family"),
                        "run_variant": summary.get("run_variant"),
                    },
                    indent=2,
                )
            )
        elif len(compare_algo_names) > 0:
            reward_parts = [
                f"{primary_label} = {primary_summary['mean_reward_40_steps']:.6f}"
            ]
            hv_summary = {}
            for algo_name in compare_algo_names:
                baseline_result = summary["baseline_results"][algo_name]
                algo_label = compare_algo_display_name(algo_name)
                reward_parts.append(
                    f"{algo_label} = {baseline_result['mean_reward_40_steps']:.6f}"
                )
                hv_summary[algo_name] = float(baseline_result["final_hv"])
            log(f"mean reward ({n_evo_steps} steps) | " + " | ".join(reward_parts))
            log(
                json.dumps(
                    {
                        "problem": summary["problem"],
                        "dim": summary["dim"],
                        "seed": summary["seed"],
                        "surrogate_model": summary["surrogate_model"],
                        "agent_name": summary["agent_name"],
                        "primary_policy": primary_policy_name,
                        "primary_inference_source": primary_inference_source,
                        "compare_algo": compare_algo_name,
                        "compare_algo_resolved": compare_algo_names,
                        "compare_inference_source": compare_inference_source,
                        "primary_final_hv": primary_summary["final_hv"],
                        "compare_final_hv": hv_summary,
                        "compare_plot_path": summary["compare_plot_path"],
                        "test_log_path": summary["test_log_path"],
                        "comparison_family": summary.get("comparison_family"),
                        "solver_family": summary.get("solver_family"),
                        "run_variant": summary.get("run_variant"),
                    },
                    indent=2,
                )
            )
        else:
            log(f"mean reward ({n_evo_steps} steps) = {summary['mean_reward_40_steps']:.6f}")
            log(
                json.dumps(
                    {
                        "problem": summary["problem"],
                        "dim": summary["dim"],
                        "seed": summary["seed"],
                        "final_hv": summary["final_hv"],
                        "final_front_size": summary["final_front_size"],
                        "wall_clock_mean_sec": summary.get("wall_clock_mean_sec"),
                        "plot_path": summary.get("plot_path"),
                    },
                    indent=2,
                )
            )
    finally:
        log_fp.close()


if __name__ == "__main__":
    main()
