import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.model_selection import train_test_split
from scipy.special import gamma
from sklearn.neighbors import NearestNeighbors

from .base import ConformalRegressor

@dataclass
class EllipsoidalGlobal(ConformalRegressor):
    """Global ellipsoidal region via Mahalanobis distance on scores."""

    alpha: float = 0.9

    # Learned parameters
    alpha_s_: Optional[float] = None
    cov_: Optional[np.ndarray] = None
    cov_inv_: Optional[np.ndarray] = None
    data_calib_mean_: Optional[np.ndarray] = None

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        test_size: Optional[float] = 0.25,
    ):
        S_cal = y_cal - y_pred_cal
        data_calib, data_valid = train_test_split(
            S_cal, test_size=test_size, random_state=42
        )
        self.data_calib_mean_ = np.mean(data_calib, axis=0)
        self.cov_ = np.cov(data_calib.T)
        self.cov_inv_ = np.linalg.inv(self.cov_)
        mahal = np.einsum("ij,ji->i", data_valid @ self.cov_inv_, data_valid.T)
        n = len(data_valid)
        self.alpha_s_ = float(
            np.quantile(mahal, np.min([np.ceil((n + 1) * self.alpha) / n, 1]))
        )
        return self

    def contains(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        if self.alpha_s_ is None:
            raise RuntimeError("EllipsoidalGlobal must be fit() before contains().")
        S_test = y_test if y_pred_test is None else (y_test - y_pred_test)
        mahal = np.einsum("ij,ji->i", S_test @ self.cov_inv_, S_test.T)
        return mahal <= self.alpha_s_

    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        mask = self.contains(X_test, y_test, y_pred_test)
        avg_coverage = float(np.mean(mask))
        S_test = y_test - y_pred_test
        d = S_test.shape[1]
        volume_unit_ball = np.pi ** (d / 2) / gamma(d / 2 + 1)
        volume_scalar = float(
            np.linalg.det(self.cov_ * self.alpha_s_) ** (1 / 2) * volume_unit_ball
        )
        volume_array = np.full(len(X_test), volume_scalar)
        return {
            "avg_coverage": avg_coverage,
            "avg_volume": volume_scalar,
            "coverage": mask,
            "volume": volume_array,
        } 





@dataclass
class EllipsoidalLocal(ConformalRegressor):
    """Local ellipsoidal baseline with KNN mixing."""

    alpha: float = 0.9
    n_neighbors: int = 100
    lam: float = 0.95

    # Learned parameters
    cov_train_: Optional[np.ndarray] = None
    knn_: Optional[object] = None
    local_alpha_s_: Optional[float] = None
    indices_split1_: Optional[np.ndarray] = None
    y_pred_cal_: Optional[np.ndarray] = None
    y_cal_: Optional[np.ndarray] = None

    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
    ):
        if y_pred_cal is None:
            raise ValueError(
                "EllipsoidalLocal.fit requires y_pred_cal to compute residuals."
            )

        self.y_cal_ = y_cal
        self.y_pred_cal_ = y_pred_cal

        indices_split1, indices_split2 = train_test_split(
            np.arange(len(X_cal)), test_size=0.5
        )
        self.indices_split1_ = indices_split1

        # on data 1
        self.knn_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.knn_.fit(X_cal[indices_split1])
        scores_cal_1 = self.y_cal_[indices_split1] - self.y_pred_cal_[indices_split1]
        self.cov_train_ = np.cov(scores_cal_1.T)

        # on data 2
        list_local_alpha_s = []
        for i in indices_split2:
            x_tick = X_cal[i].reshape(1, -1)
            s_tick = self.y_cal_[i] - self.y_pred_cal_[i]
            local_neighbors_cal = self.knn_.kneighbors(x_tick, return_distance=False)
            indices_knn = local_neighbors_cal.flatten()
            Y_local = scores_cal_1[indices_knn]
            cov_local = np.cov(Y_local.T)
            cov_mix = (1 - self.lam) * self.cov_train_ + self.lam * cov_local
            cov_mix_inv = np.linalg.inv(cov_mix)
            mahal_dist = s_tick @ cov_mix_inv @ s_tick.T
            list_local_alpha_s.append(mahal_dist)

        n2 = len(indices_split2)
        self.local_alpha_s_ = float(
            np.quantile(
                np.array(list_local_alpha_s),
                np.min([np.ceil((n2 + 1) * self.alpha) / n2, 1]),
            )
        )

        return self

    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        if self.knn_ is None:
            raise RuntimeError("EllipsoidalLocal must be fit() before metrics().")

        scores_cal_1 = (
            self.y_cal_[self.indices_split1_] - self.y_pred_cal_[self.indices_split1_]
        )
        S_test = y_test - y_pred_test

        volumes = []
        mahalanobis_dists = []
        for i in range(len(X_test)):
            x_tick = X_test[i].reshape(1, -1)
            s_tick = S_test[i]
            local_neighbors_test = self.knn_.kneighbors(x_tick, return_distance=False)
            indices_knn = local_neighbors_test.flatten()
            Y_local = scores_cal_1[indices_knn]
            cov_local = np.cov(Y_local.T)
            cov_mix = (1 - self.lam) * self.cov_train_ + self.lam * cov_local
            cov_mix_inv = np.linalg.inv(cov_mix)
            mahal_dist = s_tick @ cov_mix_inv @ s_tick.T
            mahalanobis_dists.append(mahal_dist)

            d = S_test.shape[1]
            volume_unit_ball = np.pi ** (d / 2) / gamma(d / 2 + 1)
            volume = (
                np.linalg.det(cov_mix * self.local_alpha_s_) ** (1 / 2)
                * volume_unit_ball
            )
            volumes.append(volume)

        mask = np.array(mahalanobis_dists) <= self.local_alpha_s_
        avg_coverage = float(np.mean(mask))
        avg_volume = float(np.mean(np.array(volumes)))

        return {
            "avg_coverage": avg_coverage,
            "avg_volume": avg_volume,
            "coverage": mask.astype(bool),
            "volume": np.array(volumes, dtype=float),
        }



