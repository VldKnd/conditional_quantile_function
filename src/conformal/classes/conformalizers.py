from dataclasses import dataclass

import numpy as np
import torch

from conformal.otcp.functions import learn_psi
from conformal.otcp.functions_refactor import MultivQuantileTresholdRefactor, RankFuncRefactor, sample_grid_refactor


@dataclass
class BaseRegionPredictor:
    seed: int = 0
    alpha: float = 0.1
    lower_is_better: bool = True

    def fit(self, scores_cal:np.ndarray, alpha: float, lower_is_better=True):
        pass

    def is_covered(self, scores_test) -> np.ndarray:
        return np.zeros(scores_test.shape[0], dtype=bool)


class SplitConformalPredictor(BaseRegionPredictor):

    def fit(self, scores_cal: np.ndarray, alpha: float, lower_is_better=True):
        self.alpha = alpha
        self.lower_is_better = lower_is_better
        n = len(scores_cal)
        if self.lower_is_better:
            level = np.min([np.ceil((n + 1) * (1 - alpha)) / n, 1])
        else:
            level = (n + 1) * (alpha) / n
        self.threshold = np.quantile(scores_cal, level)
        return self

    def is_covered(self, scores_test):
        if self.lower_is_better:
            coverage = scores_test <= self.threshold
        else:
            coverage = scores_test >= self.threshold
        return coverage


@dataclass
class OTCPGlobalPredictor(BaseRegionPredictor):
    split_ratio: float = 0.5

    mu: np.ndarray | None  = None
    psi: np.ndarray | None = None
    psi_star: np.ndarray | None = None

    def _solve_ot(self, scores_cal1, positive=False, seed=None):
        ''' To change the reference distribution towards a positive one, set positive = True.  '''
        # Solve OT
        self.mu = sample_grid_refactor(scores_cal1, seed=seed or self.seed, positive=positive)
        self.psi, self.psi_star = learn_psi(self.mu, scores_cal1)

    def _get_1d_scores(self, scores):
        Ranks_data = RankFuncRefactor(scores, self.mu, self.psi)
        Norm_ranks = np.linalg.norm(Ranks_data, axis=1, ord=2)
        return Norm_ranks

    def _compute_threshold(self, scores_cal2, alpha):
        # QUANTILE TRESHOLDS
        n = len(scores_cal2)
        Norm_ranks = self._get_1d_scores(scores_cal2)
        self.threshold = np.quantile(
            Norm_ranks, np.min([np.ceil((n + 1) * (1 - alpha)) / n, 1])
        )

    def fit(self, scores_cal, alpha, seed=None):
        n = scores_cal.shape[0]
        n_cal_1 = int(self.split_ratio * n)

        scores_cal1, scores_cal2 = scores_cal[:n_cal_1], scores_cal[n_cal_1:]

        self.alpha = alpha
        if self.psi is None or self.mu is None:
            self._solve_ot(scores_cal1, seed=seed)
        self._compute_threshold(scores_cal2, alpha)
        return self

    def is_covered(self, scores_test, verbose=False):
        # Computing coverage on test set
        rank_1d = self._get_1d_scores(scores_test)
        if verbose:
            print(f"{len(np.unique(rank_1d))=}")
        is_covered_otcp = rank_1d <= self.threshold
        return is_covered_otcp
