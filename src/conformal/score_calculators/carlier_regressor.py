from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from conformal.real_datasets.reproducible_split import DatasetSplit
from conformal.score_calculators.protocol import ScoreCalculator
from pushforward_operators import LinearQuantileRegression

_DEFAULT_SOLVER_ARGUMENTS = dict(
    T=5,
    epsilon=0.5,
    num_epochs=100,
    lr=0.9,
    lr_max_steps=10,
    lr_factor=0.5,
    lr_patience=100,
    lr_threshold=5 * 0.01,
    verbose=True,
    nn_init=None,
    batchsize_y=256,
    batchsize_u=256,
    inference_batch_size=256,
    full_precision=True,
    gpu=False,
    device_num=None,
    post_iter_callback=None,
)

def _make_xy_dataloader(
    X: np.ndarray, Y: np.ndarray, batch_size: int, dtype=torch.float32
) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=dtype), torch.tensor(Y, dtype=dtype))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader

@dataclass
class CarlierRegressor(ScoreCalculator):
    dtype: torch.dtype = torch.float32
    response_dimension: int | None = None
    vector_quantile_regression_solver_arguments: dict = field(
        default_factory=lambda: _DEFAULT_SOLVER_ARGUMENTS.copy()
    )
    model: LinearQuantileRegression = field(init=False)

    def __post_init__(self):
        self.model = LinearQuantileRegression(
            vector_quantile_regression_solver_arguments=self.
            vector_quantile_regression_solver_arguments
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.response_dimension = Y.shape[1]
        dataloader = _make_xy_dataloader(
            X, Y, batch_size=256, dtype=self.dtype
        )
        self.model.train()
        self.model.fit(dataloader)
        self.model.eval()

    def predict_mean(self, X: np.ndarray):
        n = X.shape[0]
        if self.response_dimension is None:
            raise RuntimeError("response_dimension is not set. Call fit() first.")
        U = torch.zeros((n, self.response_dimension), dtype=self.dtype)
        X_tensor = torch.tensor(X, dtype=self.dtype)
        self.model.eval()
        Y = self.model.push_u_given_x(x=X_tensor, u=U)
        return Y.numpy(force=True)

    def predict_quantile(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=self.dtype)
        Y = torch.tensor(Y, dtype=self.dtype)
        u = self.model.push_y_given_x(y=Y, x=X).numpy(force=True)
        self.model.eval()
        return u

    def predict_inverse_quantile(self, X: np.ndarray, U: np.ndarray):
        y = self.model.push_u_given_x(
            u=torch.tensor(U, dtype=self.dtype), x=torch.tensor(X, dtype=self.dtype)
        ).numpy(force=True)
        self.model.eval()
        return y


    def calculate_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Calculate Monge-Kantorovich qunatiles and ranks for given sample pairs (x_i, y_i).
        Additionaly, calculate estimate of the log-density fro the given samples
        Returns: quantiles, ranks, log_p

        """
        self.model.eval()
        quantiles = self.predict_quantile(X, Y) - 1/2
        ranks = np.linalg.norm(quantiles, axis=-1, ord=1)

        return {"MK Quantile": quantiles, "Cube Rank": ranks}

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        ckpt_path = path / f"model_{str(cls.__name__)}.pth"

        return cls._train_or_load(
            save_path=ckpt_path,
            model_config={},
            X_train=dataset_split.X_train,
            Y_train=dataset_split.Y_train
        )

    @classmethod
    def _train_or_load(
        cls,
        save_path: Path,
        model_config: dict,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> Self:
        print(f"{save_path=}")
        reg_cvqr = cls(**model_config)
        reg_cvqr.response_dimension = Y_train.shape[1]
        # Fit base models
        if Path.is_file(save_path):
            #reg_cvqr.model.load(trained_model_path_cvqr)
            reg_cvqr.model = LinearQuantileRegression.load_class(str(save_path))
        else:
            reg_cvqr.fit(X_train, Y_train)
            reg_cvqr.model.save(str(save_path))
        return reg_cvqr
