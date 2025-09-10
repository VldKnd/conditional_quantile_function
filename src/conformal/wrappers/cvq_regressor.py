import numpy as np
from scipy.stats import multivariate_normal
import torch
from torch.func import hessian
from torch.utils.data import TensorDataset, DataLoader

from infrastructure.classes import TrainParameters
from pushforward_operators import NeuralQuantileRegression, AmortizedNeuralQuantileRegression


def _make_xy_dataloader(X: np.ndarray, Y: np.ndarray, batch_size: int, dtype=torch.float64) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=dtype), torch.tensor(Y, dtype=dtype))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


class CVQRegressor:

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        batch_size: int,
        n_epochs: int,
        learning_rate: float = 0.01,
        dtype=torch.float64,
    ):
        self.feature_dimension: int = feature_dimension
        self.response_dimension: int = response_dimension
        self.hidden_dimension: int = hidden_dimension
        self.number_of_hidden_layers: int = number_of_hidden_layers
        self.batch_size: int = batch_size
        self.n_epochs: int = n_epochs
        self.learning_rate: float = learning_rate
        self.dtype = dtype
        self.model = AmortizedNeuralQuantileRegression(
            feature_dimension=self.feature_dimension,
            response_dimension=self.response_dimension,
            hidden_dimension=self.hidden_dimension,
            number_of_hidden_layers=self.number_of_hidden_layers,
            potential_to_estimate_with_neural_network="y",
        ).to(self.dtype)

    def fit(self, X: np.ndarray, Y:np.ndarray):
        dataloader = _make_xy_dataloader(X, Y, batch_size=self.batch_size, dtype=self.dtype)
        train_parameters = TrainParameters(number_of_epochs_to_train=self.n_epochs, 
                                           optimizer_parameters=dict(lr=self.learning_rate),
                                           scheduler_parameters={"eta_min": 0.},
                                           verbose=True)
        self.model.fit(dataloader, train_parameters=train_parameters,)

    def predict_mean(self, X: np.ndarray):
        n = X.shape[0]
        U = torch.zeros((n, self.response_dimension), dtype=self.dtype)
        X_tensor = torch.tensor(X, dtype=self.dtype)
        #dataset = TensorDataset(X_tensor, U)
        #dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        Y = self.model.push_u_given_x(x=X_tensor, u=U)
        return Y.numpy(force=True)
    
    def predict_quantile(self, X: np.ndarray, Y: np.ndarray):
        return self.model.push_y_given_x(y=torch.tensor(Y), 
                                         x=torch.tensor(X)).numpy(force=True)
    
    def predict_logdet_hessian_potential(self, X, Y, chunk_size=4096):
        _compute_batch_hessian = torch.vmap(func=hessian(self.model.potential_network, argnums=1), 
                                            in_dims=0, chunk_size=chunk_size)
        hessians = _compute_batch_hessian(torch.tensor(X), torch.tensor(Y))[:, 0].detach()
        logdet_h = torch.logdet(hessians).squeeze().detach().numpy(force=True)
        return logdet_h


def calculate_scores_cvqr(regressor: CVQRegressor, X: np.ndarray, Y: np.ndarray):
    """
    Calculate Monge-Kantorovich qunatiles and ranks for given sample pairs (x_i, y_i).
    Additionaly, calculate estimate of the log-density fro the given samples
    Returns: quantiles, ranks, log_p

    """
    n, m = X.shape
    _, d = Y.shape
    quantiles = regressor.predict_quantile(X, Y)
    ranks = np.linalg.norm(quantiles, axis=-1)
    log_p_U = multivariate_normal.logpdf(quantiles, mean=np.zeros(d))
    logdet_h = regressor.predict_logdet_hessian_potential(X, Y)
    return quantiles, ranks, log_p_U + logdet_h
