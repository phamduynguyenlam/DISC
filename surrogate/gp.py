from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF

from surrogate.surrogate_model import SurrogateModel


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    out = np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=np.float64)
    mask = np.abs(den) > float(eps)
    np.divide(num, den, out=out, where=mask)
    return out


def _suppress_gp_warnings() -> None:
    def warn(*args, **kwargs):
        return None

    import warnings

    warnings.warn = warn


@dataclass
class GPSurrogateModel(SurrogateModel):
    n_var: int
    n_obj: int
    nu: float = 5.0
    gps: list[GaussianProcessRegressor] = field(init=False, repr=False)

    @staticmethod
    def _effective_matern_nu(nu: float) -> float:
        nu_value = float(nu)
        if np.isclose(nu_value, 1.0):
            return 0.5
        if np.isclose(nu_value, 3.0):
            return 1.5
        if np.isclose(nu_value, 5.0):
            return 2.5
        return nu_value

    def __post_init__(self) -> None:
        super().__init__()
        _suppress_gp_warnings()
        self.n_var = int(self.n_var)
        self.n_obj = int(self.n_obj)
        self.nu = float(self.nu)
        effective_nu = self._effective_matern_nu(self.nu)
        self.gps = []

        def constrained_optimization(obj_func, initial_theta, bounds):
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds)
            return opt_res.x, opt_res.fun

        for _ in range(self.n_obj):
            if effective_nu > 0:
                main_kernel = Matern(
                    length_scale=np.ones(self.n_var),
                    length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)),
                    nu=effective_nu,
                )
            else:
                main_kernel = RBF(
                    length_scale=np.ones(self.n_var),
                    length_scale_bounds=(np.sqrt(1e-3), np.sqrt(1e3)),
                )

            kernel = (
                ConstantKernel(constant_value=1.0, constant_value_bounds=(np.sqrt(1e-3), np.sqrt(1e3)))
                * main_kernel
                + ConstantKernel(constant_value=1e-2, constant_value_bounds=(np.exp(-6), np.exp(0)))
            )
            self.gps.append(GaussianProcessRegressor(kernel=kernel, optimizer=constrained_optimization))

    @property
    def models(self) -> list[GaussianProcessRegressor]:
        return self.gps

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GPSurrogateModel":
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if x_arr.shape[1] != self.n_var:
            raise ValueError(f"x must have {self.n_var} variables, got {x_arr.shape[1]}.")
        if y_arr.shape[1] != self.n_obj:
            raise ValueError(f"y must have {self.n_obj} objectives, got {y_arr.shape[1]}.")

        for i, gp in enumerate(self.gps):
            gp.fit(x_arr, y_arr[:, i])
        return self

    def evaluate(
        self,
        x: np.ndarray,
        std: bool = False,
        calc_gradient: bool = False,
        calc_hessian: bool = False,
    ) -> dict[str, np.ndarray | None]:
        x_arr = np.asarray(x, dtype=np.float64)
        f_list, df_list, hf_list = [], [], []
        s_list, ds_list, hs_list = [], [], []

        for gp in self.gps:
            k = gp.kernel_(x_arr, gp.X_train_)
            y_mean = k.dot(gp.alpha_)
            f_list.append(y_mean)

            y_var = None
            y_std = None
            k_inv = None
            if std:
                l_inv = solve_triangular(gp.L_.T, np.eye(gp.L_.shape[0]))
                k_inv = l_inv.dot(l_inv.T)
                y_var = gp.kernel_.diag(x_arr)
                y_var -= np.einsum("ij,ij->i", np.dot(k, k_inv), k)
                y_var[y_var < 0] = 0.0
                y_std = np.sqrt(y_var)
                s_list.append(y_std)

            if not (calc_gradient or calc_hessian):
                continue

            ell = np.exp(gp.kernel_.theta[1:-1])
            sf2 = np.exp(gp.kernel_.theta[0])
            d = np.expand_dims(cdist(x_arr / ell, gp.X_train_ / ell), 2)
            x_exp = np.expand_dims(x_arr, 1)
            x_train_exp = np.expand_dims(gp.X_train_, 0)
            dd_n = x_exp - x_train_exp
            dd_d = d * ell**2
            dd = safe_divide(dd_n, dd_d)

            effective_nu = self._effective_matern_nu(self.nu)

            if np.isclose(effective_nu, 0.5):
                d_k = -sf2 * np.exp(-d) * dd
            elif np.isclose(effective_nu, 1.5):
                d_k = -3 * sf2 * np.exp(-np.sqrt(3) * d) * d * dd
            elif np.isclose(effective_nu, 2.5):
                d_k = -5.0 / 3.0 * sf2 * np.exp(-np.sqrt(5) * d) * (1 + np.sqrt(5) * d) * d * dd
            else:
                d_k = -sf2 * np.exp(-0.5 * d**2) * d * dd

            d_k_t = d_k.transpose(0, 2, 1)

            dy_var = None
            dy_std = None
            k_ki = None
            dk_ki = None
            if calc_gradient:
                dy_mean = d_k_t @ gp.alpha_
                df_list.append(dy_mean)

                if std and k_inv is not None and y_std is not None:
                    k_exp = np.expand_dims(k, 1)
                    k_ki = k_exp @ k_inv
                    dk_ki = d_k_t @ k_inv
                    dy_var = -np.sum(dk_ki * k_exp + k_ki * d_k_t, axis=2)
                    if np.min(y_std) != 0:
                        dy_std = 0.5 * dy_var / np.expand_dims(y_std, 1)
                    else:
                        dy_std = np.zeros(dy_var.shape, dtype=np.float64)
                    ds_list.append(dy_std)

            if calc_hessian:
                d_exp = np.expand_dims(d, 3)
                dd_exp = np.expand_dims(dd, 2)
                hd_n = d_exp * np.expand_dims(np.eye(len(ell)), (0, 1)) - np.expand_dims(x_exp - x_train_exp, 3) * dd_exp
                hd_d = d_exp**2 * np.expand_dims(ell**2, (0, 1, 3))
                hd = safe_divide(hd_n, hd_d)

                if np.isclose(effective_nu, 0.5):
                    h_k = -sf2 * np.exp(-d_exp) * (hd - dd_exp**2)
                elif np.isclose(effective_nu, 1.5):
                    h_k = -3 * sf2 * np.exp(-np.sqrt(3) * d_exp) * (d_exp * hd + (1 - np.sqrt(3) * d_exp) * dd_exp**2)
                elif np.isclose(effective_nu, 2.5):
                    h_k = (
                        -5.0
                        / 3.0
                        * sf2
                        * np.exp(-np.sqrt(5) * d_exp)
                        * (-5 * d_exp**2 * dd_exp**2 + (1 + np.sqrt(5) * d_exp) * (dd_exp**2 + d_exp * hd))
                    )
                else:
                    h_k = -sf2 * np.exp(-0.5 * d_exp**2) * ((1 - d_exp**2) * dd_exp**2 + d_exp * hd)

                h_k_t = h_k.transpose(0, 2, 3, 1)
                hy_mean = h_k_t @ gp.alpha_
                hf_list.append(hy_mean)

                if std and k_inv is not None and y_std is not None and y_var is not None and dy_var is not None and dy_std is not None:
                    k_exp = np.expand_dims(k, 2)
                    d_k_exp = np.expand_dims(d_k_t, 2)
                    dk_ki_exp = np.expand_dims(dk_ki, 2)
                    hk_ki = h_k_t @ k_inv
                    hy_var = -np.sum(hk_ki * k_exp + 2 * dk_ki_exp * d_k_exp + k_ki * h_k_t, axis=3)
                    hy_std = 0.5 * safe_divide(hy_var * y_std.reshape(-1, 1, 1) - dy_var * dy_std, y_var.reshape(-1, 1, 1))
                    hs_list.append(hy_std)

        f = np.stack(f_list, axis=1).astype(np.float32)
        df = np.stack(df_list, axis=1).astype(np.float32) if calc_gradient else None
        hf = np.stack(hf_list, axis=1).astype(np.float32) if calc_hessian else None
        s = np.stack(s_list, axis=1).astype(np.float32) if std else None
        ds = np.stack(ds_list, axis=1).astype(np.float32) if std and calc_gradient else None
        hs = np.stack(hs_list, axis=1).astype(np.float32) if std and calc_hessian else None
        return {"F": f, "dF": df, "hF": hf, "S": s, "dS": ds, "hS": hs}

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        return self.evaluate(x, std=False)["F"]

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        out = self.evaluate(x, std=True)["S"]
        if out is None:
            raise RuntimeError("Failed to compute GP predictive std.")
        return out + 1e-6


def fit_gp_surrogates(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    seed: int = 0,
    nu: float = 5.0,
) -> SurrogateModel:
    del seed
    x_arr = np.asarray(archive_x, dtype=np.float64)
    y_arr = np.asarray(archive_y, dtype=np.float64)
    model = GPSurrogateModel(n_var=int(x_arr.shape[1]), n_obj=int(y_arr.shape[1]), nu=float(nu))
    return model.fit(x_arr, y_arr)
