from typing import Self
from pathlib import Path

import numpy as np

from conformal.real_datasets.reproducible_split import DatasetSplit


class ScoreCalculator:

    def calculate_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        return {"Zero": np.zeros_like(Y)}

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        return cls()
