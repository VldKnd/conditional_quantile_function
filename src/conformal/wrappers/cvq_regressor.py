from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.func import hessian
from torch.utils.data import TensorDataset, DataLoader

from infrastructure.classes import TrainParameters
from pushforward_operators.convex_potential_flow.core_flow import ConvexPotentialFlow
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators import NeuralQuantileRegression, AmortizedNeuralQuantileRegression


def _make_xy_dataloader(
    X: np.ndarray, Y: np.ndarray, batch_size: int, dtype=torch.float64
) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=dtype), torch.tensor(Y, dtype=dtype))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


class ScoreCalculator:

    def calculate_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        return {"Zero": np.zeros_like(Y)}


@dataclass
class BaseVQRegressor(ScoreCalculator):

    feature_dimension: int
    response_dimension: int
    hidden_dimension: int
    number_of_hidden_layers: int
    batch_size: int
    n_epochs: int
    learning_rate: float = 0.01
    dtype: torch.dtype = torch.float64
    betas: tuple[float, float] = (0.5, 0.5)
    weight_decay: float = 1e-4
    warmup_iterations: int = 5
    model: PushForwardOperator = field(init=False)

    def __post_init__(self):
        _optimizer_parameters = dict(
            lr=self.learning_rate, betas=self.betas, weight_decay=self.weight_decay
        )
        self.train_parameters = TrainParameters(
            number_of_epochs_to_train=self.n_epochs,
            optimizer_parameters=_optimizer_parameters,
            scheduler_parameters={"eta_min": 0.},
            verbose=True,
            warmup_iterations=self.warmup_iterations,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        dataloader = _make_xy_dataloader(
            X, Y, batch_size=self.batch_size, dtype=self.dtype
        )
        self.model.fit(
            dataloader,
            train_parameters=self.train_parameters,
        )

    def predict_mean(self, X: np.ndarray):
        n = X.shape[0]
        U = torch.zeros((n, self.response_dimension), dtype=self.dtype)
        X_tensor = torch.tensor(X, dtype=self.dtype)
        #dataset = TensorDataset(X_tensor, U)
        #dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        Y = self.model.push_u_given_x(x=X_tensor, u=U)
        return Y.numpy(force=True)

    def predict_quantile(self, X: np.ndarray, Y: np.ndarray):
        return self.model.push_y_given_x(
            y=torch.tensor(Y, dtype=self.dtype), x=torch.tensor(X, dtype=self.dtype)
        ).numpy(force=True)


@dataclass
class CVQRegressor(BaseVQRegressor):

    def __post_init__(self):
        super().__post_init__()
        self.model = AmortizedNeuralQuantileRegression(
            feature_dimension=self.feature_dimension,
            response_dimension=self.response_dimension,
            hidden_dimension=self.hidden_dimension,
            number_of_hidden_layers=self.number_of_hidden_layers,
            potential_to_estimate_with_neural_network="y",
        ).to(self.dtype)

    def predict_logdet_hessian_potential(self, X, Y, batch_size: int | None = None):
        _compute_batch_hessian = torch.vmap(
            func=hessian(self.model.potential_network, argnums=1),
            in_dims=0,
            chunk_size=batch_size or self.batch_size
        )
        hessians = _compute_batch_hessian(
            torch.tensor(X, dtype=self.dtype), torch.tensor(Y, dtype=self.dtype)
        )[:, 0].detach()
        logdet_h = torch.logdet(hessians).squeeze().detach().numpy(force=True)
        return logdet_h

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
        n, m = X.shape
        _, d = Y.shape
        quantiles = self.predict_quantile(X, Y)
        ranks = np.linalg.norm(quantiles, axis=-1)
        log_p_U = multivariate_normal.logpdf(quantiles, mean=np.zeros(d))
        logdet_h = self.predict_logdet_hessian_potential(X, Y, batch_size=batch_size)
        return {
            "MK Quantile": quantiles,
            "MK Rank": ranks,
            "Log Density": log_p_U + logdet_h
        }


@dataclass
class CPFlowRegressor(BaseVQRegressor):
    n_blocks: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.model = ConvexPotentialFlow(
            feature_dimension=self.feature_dimension,
            response_dimension=self.response_dimension,
            hidden_dimension=self.hidden_dimension,
            number_of_hidden_layers=self.number_of_hidden_layers,
            n_blocks=self.n_blocks,
        ).to(self.dtype)

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
        n, m = X.shape
        _, d = Y.shape

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantiles = self.predict_quantile(X, Y)
            ranks = np.linalg.norm(quantiles, axis=-1)
            log_p = self.model.logp_cond(
                Y=torch.tensor(Y, dtype=self.dtype),
                X=torch.tensor(X, dtype=self.dtype)
            ).numpy(force=True)

        return {"MK Quantile": quantiles, "MK Rank": ranks, "Log Density": log_p}
