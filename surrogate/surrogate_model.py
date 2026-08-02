from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


def surrogate_model_name(args: Any) -> str:
    name = str(getattr(args, "surrogate_model", "gp")).strip().lower()
    if name != "gp":
        raise ValueError(f"Unsupported surrogate model: {name}. Only 'gp' is available.")
    return name


class SurrogateModel(ABC):
    @abstractmethod
    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        raise NotImplementedError

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def estimate_uncertainty(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    archive_pred: np.ndarray,
    offspring_x: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    archive_pred = np.asarray(archive_pred, dtype=np.float32)
    offspring_x = np.asarray(offspring_x, dtype=np.float32)
    residual = np.abs(archive_pred - archive_y)
    neighbor_count = min(int(n_neighbors), int(archive_x.shape[0]))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    neighbors.fit(archive_x)
    neighbor_indices = neighbors.kneighbors(offspring_x, return_distance=False)
    return residual[neighbor_indices].mean(axis=1).astype(np.float32) + 1e-6


from surrogate.gp import (  # noqa: E402
    GPSurrogateModel,
    fit_gp_surrogates,
)


surrogate_model = SurrogateModel
gp = GPSurrogateModel


__all__ = [
    "GPSurrogateModel",
    "SurrogateModel",
    "estimate_uncertainty",
    "fit_gp_surrogates",
    "surrogate_model_name",
]
