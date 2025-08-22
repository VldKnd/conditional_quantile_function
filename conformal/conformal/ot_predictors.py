import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split
import ot
from scipy.stats import qmc, norm
from sklearn.neighbors import NearestNeighbors

from .base import ConformalRegressor


def sample_reference_grid(data_like: np.ndarray, positive: bool = False) -> np.ndarray:
    """
    Sample a reference grid of rank vectors U_i used for MK ranks.

    Math:
    - Draw radii R_i = i/n in [0,1].
    - If positive=False (center-outward order on R^d): sample Z_i ~ N(0,I_d) and set U_i = R_i Z_i/||Z_i||.
    - If positive=True (left-to-right order on R_+^d): sample Z_i ~ Dirichlet(1,...,1) and set U_i = R_i Z_i.

    Args:
    - data_like: shape (n, d), used only to infer n and d.
    - positive: whether to restrict to the positive orthant.

    Returns:
    - U: array of shape (n, d) of reference rank vectors.
    """
    n = data_like.shape[0]
    d = data_like.shape[1]
    radii = np.linspace(0, 1, n)
    if not positive:
        sampler = qmc.Halton(d=d)
        gaussian = sampler.random(n=n + 1)[1:]
        gaussian = norm.ppf(gaussian, loc=0, scale=1)
        U = [radii[i] * z / np.linalg.norm(z) for i, z in enumerate(gaussian)]
    else:
        U = [
            radii[i] * z / np.sum(z)
            for i, z in enumerate(np.random.exponential(scale=1.0, size=(n, d)))
        ]
    return np.array(U)


def learn_brenier_potentials(mu: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the discrete 2-Wasserstein OT between empirical data and reference grid.

    Math:
    - Solve the Kantorovich problem with cost c(x,y)=||x-y||^2/2, yielding dual potentials (g,f).
    - Return Brenier potentials: ψ(mu) = ||mu||^2/2 - f, and ψ*(data) = ||data||^2/2 - g.

    Args:
    - mu: reference support, shape (n, d).
    - data: empirical scores, shape (n, d).

    Returns:
    - (psi, psi_star): both arrays of shape (n,).
    """
    M = ot.dist(data, mu) / 2
    res = ot.solve(M)
    g, f = res.potentials
    psi = 0.5 * np.linalg.norm(mu, axis=1) ** 2 - f
    psi_star = 0.5 * np.linalg.norm(data, axis=1) ** 2 - g
    return psi, psi_star


def evaluate_brenier_rank(x: np.ndarray, mu: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """
    Evaluate the MK rank map R_n(x) = argmax_{u in mu} <u, x> - ψ(u).

    Math:
    - Discrete argmax over the reference grid `mu` using the convex potential ψ.

    Args:
    - x: a point (d,) or batch (n, d).
    - mu: reference grid (m, d).
    - psi: ψ evaluated on mu (m,).

    Returns:
    - The maximizer(s) in mu: (d,) or (n, d).
    """
    if len(x.shape) == 1:
        to_max = (mu @ x) - psi
        return mu[np.argmax(to_max)]
    to_max = (mu @ x.T).T - psi
    return mu[np.argmax(to_max, axis=1)]


def quantile_map(u: np.ndarray, data: np.ndarray, psi_star: np.ndarray) -> np.ndarray:
    """
    Evaluate the vector quantile map Q_n(u) = argmax_{y in data} <u, y> - ψ*(y).

    Math:
    - Discrete argmax over empirical data using the convex conjugate ψ*.

    Args:
    - u: a point (d,) or batch (n, d) in the reference domain.
    - data: empirical scores (m, d).
    - psi_star: ψ* evaluated on data (m,).

    Returns:
    - The maximizer(s) in data: (d,) or (n, d).
    """
    return evaluate_brenier_rank(u, data, psi_star)


def estimate_quantile_region_volume(
    scores: np.ndarray,
    mu: np.ndarray,
    psi: np.ndarray,
    quantile_threshold: float,
    N: int = int(1e6),
) -> float:
    """
    Monte Carlo estimate of the volume of the MK quantile region {s: ||R_n(s)|| <= q}.

    Math:
    - Sample v uniformly in the bounding box of `scores` (importance-free MCMC over a box).
    - Estimate Vol ≈ Vol(box) * P(||R_n(v)|| <= q).

    Args:
    - scores: calibration-like sample to bound the region (n, d).
    - mu: reference grid (m, d).
    - psi: ψ(mu) values (m,).
    - quantile_threshold: q ≥ 0.
    - N: number of Monte Carlo samples.

    Returns:
    - Estimated volume (float).
    """
    M = np.max(scores, axis=0)
    m = np.min(scores, axis=0)
    v = m + np.random.random((N, mu.shape[1])) * (M - m)
    scale = np.prod(M - m)
    mcmc = np.mean(np.linalg.norm(evaluate_brenier_rank(v, mu, psi), axis=1) <= quantile_threshold)
    return float(mcmc * scale)


@dataclass
class OTCPGlobal(ConformalRegressor):
    """Global MK quantile region calibrated on scores, independent of X.

    Math:
    - Scores S = Y - f̂(X).
    - Learn (mu, ψ, ψ*) on a calibration split of S, compute MK rank R and threshold q from ||R||.
    - Inclusion at test: ||R(S_test)|| <= q.
    """

    alpha: float = 0.9

    # Learned parameters
    quantile_threshold_: Optional[float] = None
    mu_: Optional[np.ndarray] = None
    psi_: Optional[np.ndarray] = None
    psi_star_: Optional[np.ndarray] = None
    data_calib_: Optional[np.ndarray] = None

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
    ):
        S_cal = y_cal if y_pred_cal is None else (y_cal - y_pred_cal)
        data_calib, data_valid = train_test_split(S_cal, test_size=0.25)

        self.mu_ = sample_reference_grid(data_calib)
        self.psi_, self.psi_star_ = learn_brenier_potentials(self.mu_, data_calib)

        n = len(data_valid)
        ranks_data_valid = evaluate_brenier_rank(data_valid, self.mu_, self.psi_)
        norm_ranks_valid = np.linalg.norm(ranks_data_valid, axis=1, ord=2)

        self.quantile_threshold_ = float(
            np.quantile(
                norm_ranks_valid, np.min([np.ceil((n + 1) * self.alpha) / n, 1])
            )
        )
        self.data_calib_ = data_calib
        return self

    def contains(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if self.quantile_threshold_ is None:
            raise RuntimeError("OTCPGlobal must be fit() before contains().")
        S_test = y_test - y_pred_test
        R = evaluate_brenier_rank(S_test, self.mu_, self.psi_)
        mask = np.linalg.norm(R, axis=1) <= self.quantile_threshold_
        return mask

    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        mask = self.contains(X_test, y_test, y_pred_test)
        avg_coverage = float(np.mean(mask))
        S_any = y_test if y_pred_test is None else (y_test - y_pred_test)
        volume_scalar = estimate_quantile_region_volume(
            S_any, self.mu_, self.psi_, self.quantile_threshold_
        )
        volume_array = np.full(len(X_test), volume_scalar)
        return {
            "avg_coverage": avg_coverage,
            "avg_volume": volume_scalar,
            "coverage": mask,
            "volume": volume_array,
        }


@dataclass
class OTCPAdaptiveKNN(ConformalRegressor):
    """Adaptive OT-CP+ with KNN neighborhoods in X.

    Math:
    - Split calibration X into two halves; fit KNN on one half. For each held-out point, use KNN neighbors' scores to solve local OT and compute ||R||.
    - Calibrate q at level alpha from ||R|| on held-out points.
    - At test x, solve local OT on x's KNN neighbors and include y iff ||R(y- f̂(x))|| <= q.
    """

    alpha: float = 0.9
    n_neighbors: int = 100

    # Learned parameters
    quantile_threshold_: Optional[float] = None
    knn_: Optional[object] = None
    scores_cal_1_: Optional[np.ndarray] = None
    mu_: Optional[np.ndarray] = None
    otcp_global_: Optional[OTCPGlobal] = None

    def _fit_knn(self, X_cal):
        self.knn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn_.fit(X_cal)

    def _get_local_ot_params(self, x_tick):
        local_neighbors = self.knn_.kneighbors(
            x_tick.reshape(1, -1), return_distance=False
        )
        indices_knn = local_neighbors.flatten()
        Y_local = self.scores_cal_1_[indices_knn]

        # Ensure mu is initialized for the local OT problem
        if not hasattr(self, "mu_") or self.mu_ is None:
            self.mu_ = sample_reference_grid(
                np.zeros((self.n_neighbors, Y_local.shape[1]))
            )

        psi, psi_star = learn_brenier_potentials(self.mu_, Y_local)
        return psi, Y_local

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
    ):
        S_cal = y_cal if y_pred_cal is None else (y_cal - y_pred_cal)
        indices_split1, indices_split2 = train_test_split(
            np.arange(len(X_cal)), test_size=0.5
        )

        X_cal_1 = X_cal[indices_split1]
        self.scores_cal_1_ = S_cal[indices_split1]
        self._fit_knn(X_cal_1)

        self.otcp_global_ = OTCPGlobal()  # To access OT methods
        self.mu_ = sample_reference_grid(
            np.zeros((self.n_neighbors, S_cal.shape[1]))
        )

        list_MK_ranks = []
        for i in indices_split2:
            x_tick = X_cal[i]
            s_tick = S_cal[i]
            psi_local, _ = self._get_local_ot_params(x_tick)
            rank = evaluate_brenier_rank(s_tick, self.mu_, psi_local)
            list_MK_ranks.append(rank)

        n = len(indices_split2)
        norm_ranks = np.linalg.norm(np.array(list_MK_ranks), axis=1)
        self.quantile_threshold_ = float(
            np.quantile(norm_ranks, np.min([np.ceil((n + 1) * self.alpha) / n, 1]))
        )
        return self

    def contains(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if self.quantile_threshold_ is None:
            raise RuntimeError("OTCPAdaptiveKNN must be fit() before contains().")
        S_test = y_test if y_pred_test is None else (y_test - y_pred_test)
        masks = []
        for i in range(len(X_test)):
            psi_local, _ = self._get_local_ot_params(X_test[i])
            R = evaluate_brenier_rank(S_test[i], self.mu_, psi_local)
            masks.append(np.linalg.norm(R) <= self.quantile_threshold_)
        return np.array(masks, dtype=bool)

    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        if self.quantile_threshold_ is None:
            raise RuntimeError("OTCPAdaptiveKNN must be fit() before metrics().")
        S_test = y_test - y_pred_test

        ranks = []
        volumes = []
        for i in range(len(X_test)):
            psi_local, Y_local = self._get_local_ot_params(X_test[i])
            rank = evaluate_brenier_rank(S_test[i], self.mu_, psi_local)
            ranks.append(np.linalg.norm(rank))
            volume = estimate_quantile_region_volume(
                Y_local, self.mu_, psi_local, self.quantile_threshold_
            )
            volumes.append(volume)

        mask = np.array(ranks) <= self.quantile_threshold_
        avg_coverage = float(np.mean(mask))
        avg_volume = float(np.mean(volumes))
        return {
            "avg_coverage": avg_coverage,
            "avg_volume": avg_volume,
            "coverage": mask.astype(bool),
            "volume": np.array(volumes, dtype=float),
        }


@dataclass
class RectangleGlobal(ConformalRegressor):
    """Global hyperrectangle region via Bonferroni-corrected marginal quantiles."""

    alpha: float = 0.9

    # Learned parameters
    list_axis_: Optional[np.ndarray] = None

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
    ):
        S_cal = y_cal if y_pred_cal is None else (y_cal - y_pred_cal)
        d = S_cal.shape[1]
        componentwise_alpha = 1 - (1 - self.alpha) / d
        center_outward_m = (1 - componentwise_alpha) / 2
        list_axis = []
        for k in range(d):
            axis = np.quantile(S_cal.T[k], [1 - center_outward_m, center_outward_m])
            list_axis.append(axis)
        self.list_axis_ = np.array(list_axis)
        return self

    def contains(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if self.list_axis_ is None:
            raise RuntimeError("RectangleGlobal must be fit() before contains().")
        S_test = y_test if y_pred_test is None else (y_test - y_pred_test)
        within_lower_bounds = np.all(self.list_axis_.T[1] <= S_test, axis=1)
        within_upper_bounds = np.all(S_test <= self.list_axis_.T[0], axis=1)
        return within_lower_bounds & within_upper_bounds

    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        mask = self.contains(X_test, y_test, y_pred_test)
        avg_coverage = float(np.mean(mask))
        volume_scalar = float(np.prod(self.list_axis_.T[0] - self.list_axis_.T[1]))
        volume_array = np.full(len(X_test), volume_scalar)
        return {
            "avg_coverage": avg_coverage,
            "avg_volume": volume_scalar,
            "coverage": mask,
            "volume": volume_array,
        } 