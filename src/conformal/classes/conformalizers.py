from dataclasses import dataclass, field

import numpy as np
import torch

from tqdm.auto import tqdm

from conformal.otcp.ellipsoidal_conformal_utilities import ellipse_local_alpha_s, ellipse_volume, ellipsoidal_non_conformity_measure
from conformal.otcp.functions import ConditionalRank_Adaptive, MultivQuantileTreshold_Adaptive, get_volume_QR, learn_psi
from conformal.otcp.functions_refactor import MultivQuantileTresholdRefactor, RankFuncRefactor, sample_grid_refactor


@dataclass
class BaseRegionPredictor:
    seed: int = 0
    alpha: float = 0.1
    lower_is_better: bool = True

    def fit(self, X_cal: np.ndarray, scores_cal: np.ndarray, alpha: float):
        pass

    def is_covered(
        self,
        X_test: np.ndarray,
        scores_test: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        return np.zeros(scores_test.shape[0], dtype=bool)


class SplitConformalPredictor(BaseRegionPredictor):

    def fit(self, X_cal: np.ndarray, scores_cal: np.ndarray, alpha: float, lower_is_better=True):
        self.alpha = alpha
        self.lower_is_better = lower_is_better
        n = len(scores_cal)
        if self.lower_is_better:
            level = np.min([np.ceil((n + 1) * (1 - alpha)) / n, 1])
        else:
            level = (n + 1) * (alpha) / n
        self.threshold = np.quantile(scores_cal, level)
        return self

    def is_covered(
        self,
        X_test: np.ndarray,
        scores_test: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        if self.lower_is_better:
            coverage = scores_test <= self.threshold
        else:
            coverage = scores_test >= self.threshold
        return coverage


@dataclass
class OTCPGlobalPredictor(BaseRegionPredictor):
    split_ratio: float = 0.5

    mu: np.ndarray = field(init=False, default_factory=lambda : np.zeros(0))
    psi: np.ndarray = field(init=False, default_factory=lambda : np.zeros(0))
    psi_star: np.ndarray = field(init=False, default_factory=lambda : np.zeros(0))

    def _solve_ot(
        self, scores_cal1: np.ndarray, positive: bool = False, seed: int | None = None
    ):
        ''' To change the reference distribution towards a positive one, set positive = True.  '''
        # Solve OT
        self.mu = sample_grid_refactor(
            scores_cal1, seed=seed or self.seed, positive=positive
        )
        self.psi, self.psi_star = learn_psi(self.mu, scores_cal1)

    def _get_1d_scores(self, scores: np.ndarray) -> np.ndarray:
        Ranks_data = RankFuncRefactor(scores, self.mu, self.psi)
        Norm_ranks = np.linalg.norm(Ranks_data, axis=1, ord=2)
        return Norm_ranks

    def _compute_threshold(self, scores_cal2: np.ndarray, alpha: float):
        # QUANTILE TRESHOLDS
        n = len(scores_cal2)
        Norm_ranks = self._get_1d_scores(scores_cal2)
        self.threshold = np.quantile(
            Norm_ranks, np.min([np.ceil((n + 1) * (1 - alpha)) / n, 1])
        )

    def fit(
        self,
        X_cal: np.ndarray,
        scores_cal: np.ndarray,
        alpha: float,
        seed: int | None = None
    ):
        n = scores_cal.shape[0]
        n_cal_1 = int(self.split_ratio * n)

        scores_cal1, scores_cal2 = scores_cal[:n_cal_1], scores_cal[n_cal_1:]

        self.alpha = alpha
        if len(self.psi) == 0 or len(self.mu) == 0:
            self._solve_ot(scores_cal1, seed=seed)
        self._compute_threshold(scores_cal2, alpha)
        return self

    def is_covered(
        self,
        X_test: np.ndarray,
        scores_test: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        # Computing coverage on test set
        rank_1d = self._get_1d_scores(scores_test)
        if verbose:
            print(f"{len(np.unique(rank_1d))=}")
        is_covered_otcp = rank_1d <= self.threshold
        return is_covered_otcp


@dataclass
class OTCPLocalPredictor(BaseRegionPredictor):
    # Rule for choosing number of neighbors:
    # "large": sqrt(len(X_cal))
    # other: 0.1 * len(X_cal)
    knn_mode: str = "large"

    def fit(self, X_cal: np.ndarray, scores_cal: np.ndarray, alpha: float):
        self.alpha = alpha
        if self.knn_mode == "large":
            n_neighbors = int(len(X_cal)**0.5)
        else:
            n_neighbors = int(len(X_cal) * 0.1)
        self.Quantile_Treshold, self.knn, self.scores_cal_1, self.mu = MultivQuantileTreshold_Adaptive(
            scores_cal, X_cal, n_neighbors=n_neighbors, alpha=1 - alpha
        )

    def is_covered(
        self,
        X_test: np.ndarray,
        scores_test: np.ndarray,
        verbose: bool = False
    ) -> np.ndarray:
        n = self.knn.get_params()["n_neighbors"]
        n_test = X_test.shape[0]
        is_covered = np.zeros(n_test, dtype=bool)
        for i in tqdm(range(n_test), disable=not verbose):
            ConditionalRank, psi, Y = ConditionalRank_Adaptive(
                scores_test[i],
                X_test[i],
                self.knn,
                self.scores_cal_1,
                n_neighbors=n,
                mu=self.mu
            )
            is_covered[i] = np.linalg.norm(ConditionalRank) <= self.Quantile_Treshold
        return is_covered

    def get_volume(
        self, x: np.ndarray, score: np.ndarray, verbose: bool = False
    ) -> float:
        # Get volume at a single point
        n = self.knn.get_params()["n_neighbors"]
        ConditionalRank, psi, Y = ConditionalRank_Adaptive(
            score, x, self.knn, self.scores_cal_1, n_neighbors=n, mu=self.mu
        )
        return get_volume_QR(self.Quantile_Treshold, self.mu, psi, Y)


@dataclass
class EllipsoidalLocal(BaseRegionPredictor):
    split_ratio: float = 0.5
    lam: float = 0.95  # lambda parameter for mixing local and global covariance
    n_neighbors: int = 100
    det_cov: float = field(init=False)
    volume: float = field(init=False)

    def fit(self, X_cal: np.ndarray, scores_cal: np.ndarray, alpha: float):
        self.alpha = alpha
        n = scores_cal.shape[0]
        self.n_cal_1 = int(self.split_ratio * n)
        self.scores_cal_1 = scores_cal[:self.n_cal_1]
        self.scores_cal_2 = scores_cal[self.n_cal_1:]
        self.cov_cal_1 = np.cov(self.scores_cal_1.T)
        self.knn, self.local_alpha_s = ellipse_local_alpha_s(
            x_train=X_cal[:self.n_cal_1],
            x_cal=X_cal[self.n_cal_1:],
            y_true_train=self.scores_cal_1,
            y_pred_train=np.zeros_like(self.scores_cal_1),
            y_true_cal=self.scores_cal_2,
            y_pred_cal=np.zeros_like(self.scores_cal_2),
            epsilon=self.alpha,
            n_neighbors=self.n_neighbors,
            lam=self.lam,
            cov_train=self.cov_cal_1,
        )
    
    def _get_local_inv_cov(self, local_neighbors: np.ndarray) -> np.ndarray:
        knn_scores = self.scores_cal_1[
            local_neighbors, :
        ]
        local_cov_test = np.cov(knn_scores.T)
        local_cov_test_regularized = self.lam * local_cov_test + (1 - self.lam) * self.cov_cal_1
        local_inv_cov_test = np.linalg.inv(local_cov_test_regularized)

        return local_inv_cov_test
    
    def is_covered(self, X_test: np.ndarray, scores_test: np.ndarray, verbose: bool = False) -> np.ndarray:
        local_neighbors_test = self.knn.kneighbors(X_test, return_distance=False)
        is_covered = np.zeros(X_test.shape[0])
        for i in tqdm(range(X_test.shape[0]), disable=not verbose):
            local_inv_cov_test = self._get_local_inv_cov(local_neighbors_test[i, :])
            is_covered[i] = ellipsoidal_non_conformity_measure(scores_test[i], local_inv_cov_test) <= self.local_alpha_s
            
        return is_covered
    
    def get_volume(
        self, x: np.ndarray, score: np.ndarray, verbose: bool = False) -> float:
        local_neighbors = self.knn.kneighbors(x.reshape(1, -1), return_distance=False)
        local_inv_cov = self._get_local_inv_cov(local_neighbors[0, :])
        d = score.shape[-1]
        return ellipse_volume(local_inv_cov, self.local_alpha_s, d)
