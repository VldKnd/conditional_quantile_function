from typing import Self
from pathlib import Path
from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor

from conformal.real_datasets.reproducible_split import DatasetSplit
from conformal.score_calculators.protocol import ScoreCalculator


class RandomForestWithScore(RandomForestRegressor, ScoreCalculator):

    def calculate_scores(self,
                         X: ndarray,
                         Y: ndarray,
                         batch_size: int | None = None) -> dict[str, ndarray]:
        return {"Signed Error": Y - self.predict(X)}

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        rf = cls(random_state=args.seed, n_jobs=args.n_cpus)
        rf.fit(dataset_split.X_train, dataset_split.Y_train)
        return rf
