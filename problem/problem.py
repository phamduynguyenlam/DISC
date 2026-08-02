from __future__ import annotations

import numpy as np

SUPPORTED_PROBLEMS: list[str] = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]

REFERENCE_POINTS: dict[str, np.ndarray] = {
    "ZDT1": np.array([0.9994, 6.0576], dtype=np.float32),
    "ZDT2": np.array([0.9994, 6.8960], dtype=np.float32),
    "ZDT3": np.array([0.9994, 6.0571], dtype=np.float32),
    "DTLZ2": np.array([2.8390, 2.9011, 2.8575], dtype=np.float32),
    "DTLZ3": np.array([2421.6427, 1905.2767, 2532.9691], dtype=np.float32),
    "DTLZ4": np.array([3.2675, 2.6443, 2.4263], dtype=np.float32),
    "DTLZ5": np.array([2.6672, 2.8009, 2.8575], dtype=np.float32),
    "DTLZ6": np.array([16.8258, 16.9194, 17.7646], dtype=np.float32),
    "DTLZ7": np.array([0.9984, 0.9961, 22.8114], dtype=np.float32),
}

TRUE_PARETO_HV: dict[str, float] = {
    "ZDT1-30D-2obj": 5.719354044543262,
    "ZDT2-30D-2obj": 6.223946695544582,
    "ZDT3-30D-2obj": 6.094808522998477,
    "DTLZ2-30D-3obj": 22.96077571671862,
    "DTLZ3-30D-3obj": 11686863768.88893,
    "DTLZ4-30D-3obj": 20.389607896010133,
    "DTLZ5-30D-3obj": 18.6312399291,
    "DTLZ6-30D-3obj": 5038.7719154348,
    "DTLZ7-30D-3obj": 18.1549480498,
}

def default_problem_dim(name: str) -> int:
    return 30


def get_reference_point(problem_name: str, *, n_obj: int | None = None) -> np.ndarray:
    key = str(problem_name).upper()
    if key not in REFERENCE_POINTS:
        raise ValueError(f"Unsupported problem for reference point: {problem_name}")
    ref = np.asarray(REFERENCE_POINTS[key], dtype=np.float32).reshape(-1)
    return ref.astype(np.float32) if n_obj is None else ref[: int(n_obj)].astype(np.float32)


def get_true_pareto_hv(problem_name: str, *, dim: int, n_obj: int) -> float | None:
    problem_key = str(problem_name).upper()
    exact_key = f"{problem_key}-{int(dim)}D-{int(n_obj)}obj"
    fallback_key = f"{problem_key}-30D-{int(n_obj)}obj"
    if exact_key in TRUE_PARETO_HV:
        return float(TRUE_PARETO_HV[exact_key])
    if fallback_key in TRUE_PARETO_HV:
        return float(TRUE_PARETO_HV[fallback_key])
    raise ValueError(
        f"Unsupported problem/dim/objective setting for true Pareto HV: {exact_key} "
        f"(no 30D fallback found at {fallback_key})"
    )


class ZDTProblem:
    """ZDT benchmark problems (minimization)."""

    def __init__(self, name: str, dim: int = 30):
        self.name = str(name).upper()
        self.dim = int(dim)
        self.lower = 0.0
        self.upper = 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")

        f1 = x[:, 0]
        g = 1.0 + 9.0 / (self.dim - 1.0) * np.sum(x[:, 1:], axis=1)

        if self.name == "ZDT1":
            h = 1.0 - np.sqrt(f1 / g)
        elif self.name == "ZDT2":
            h = 1.0 - (f1 / g) ** 2
        elif self.name == "ZDT3":
            h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        else:
            raise ValueError(f"Unsupported problem: {self.name}")

        f2 = g * h
        return np.stack([f1, f2], axis=1).astype(np.float32)


class DTLZProblem:
    """DTLZ benchmark problems (minimization)."""

    def __init__(self, name: str, dim: int = 30):
        self.name = str(name).upper()
        self.dim = int(dim)
        self.lower = 0.0
        self.upper = 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")

        if self.name == "DTLZ2":
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = 0.5 * np.pi * x[:, 1]
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ3":
            g = 100.0 * (
                self.dim - 2.0
                + np.sum((x[:, 2:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, 2:] - 0.5)), axis=1)
            )
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = 0.5 * np.pi * x[:, 1]
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ4":
            alpha = 100.0
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * (x[:, 0] ** alpha)
            theta2 = 0.5 * np.pi * (x[:, 1] ** alpha)
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ5":
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, 1])
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ6":
            g = np.sum(x[:, 2:] ** 0.1, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, 1])
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ7":
            denom = max(float(self.dim) - 2.0, 1.0)
            g = 1.0 + 9.0 * np.sum(x[:, 2:], axis=1) / denom
            f1 = x[:, 0]
            f2 = x[:, 1]
            term1 = (f1 / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * f1))
            term2 = (f2 / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * f2))
            h = 3.0 - (term1 + term2)
            f3 = (1.0 + g) * h
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        raise ValueError(f"Unsupported problem: {self.name}")


def make_problem(name: str, dim: int | None = None):
    key = str(name).upper()
    if key.startswith("ZDT"):
        return ZDTProblem(key, dim=default_problem_dim(key) if dim is None else int(dim))
    if key.startswith("DTLZ"):
        return DTLZProblem(key, dim=default_problem_dim(key) if dim is None else int(dim))
    raise ValueError(f"Unsupported problem: {name}")
