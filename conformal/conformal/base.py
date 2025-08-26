from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans


@dataclass
class ConformalRegressor(ABC):
    """Abstract base class for a conformal regressor."""

    alpha: float = 0.9

    @abstractmethod
    def fit(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        y_pred_cal: Optional[np.ndarray] = None,
    ):
        ...

    def contains(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        # A boolean valued return to measure coverage.
        return None

    @abstractmethod
    def metrics(
        self, X_test: np.ndarray, y_test: np.ndarray, y_pred_test: np.ndarray
    ) -> Dict[str, Any]:
        ...

    def worst_set_coverage(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        n_clusters: int = 10,
    ) -> Tuple[float, float]:
        k_means = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
        labels = k_means.fit_predict(X_test)

        coverages = []
        for l in np.unique(labels):
            mask = labels == l
            if np.sum(mask) == 0:
                continue

            metrics = self.metrics(X_test[mask], y_test[mask], y_pred_test[mask])
            coverages.append(metrics["avg_coverage"]) 

        min_cvg = np.min(coverages) if coverages else 1.0
        std_cvg = np.std(coverages) if coverages else 0.0
        return (float(min_cvg), float(std_cvg)) 