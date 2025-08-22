from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch

from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr import VectorQuantileRegressor
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr.solvers.regularized_lse import MLPRegularizedDualVQRSolver

class FastNonLinearVectorQuantileRegression(PushForwardOperator):
    def __init__(self, input_dimension:int, embedding_dimension: int = 5, hidden_dimension: int = 100, number_of_hidden_layers: int = 1):
        self.epsilon = 1e-5
        self.input_dimension = input_dimension
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_dimension = hidden_dimension
        self.embedding_dimension = embedding_dimension
        self.fitted = False
        self.vector_quantile_regression: VectorQuantileRegressor | None = None

    def to(self, *args, **kwargs):
        """
        Moves the model to the specified device and dtype.
        """
        if self.fitted and self.vqr is not None:
            self.vqr.to(*args, **kwargs)
            return self

    def train(self):
        """
        Sets the model to training mode.
        """
        pass

    def eval(self):
        """
        Sets the model to evaluation mode.
        """
        pass

    def fit(self, dataloader: torch.utils.data.DataLoader, *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_parameters (TrainParameters): Training parameters.
        """

        X_Y_tuple = [(X_batch, Y_batch) for X_batch, Y_batch in dataloader]
        X_tensor = torch.cat([X_batch for X_batch, _ in X_Y_tuple], dim=0)
        Y_tensor = torch.cat([Y_batch for _, Y_batch in X_Y_tuple], dim=0)
        self.fit_tensor(X_tensor=X_tensor, Y_tensor=Y_tensor, *args, **kwargs)
        return self

    def fit_tensor(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
            verbose (bool): Whether to print verbose output.
        """
        vector_quantile_regression_solver = MLPRegularizedDualVQRSolver(
            hidden_layers = (
                self.input_dimension,
                *[self.hidden_dimension]*self.number_of_hidden_layers,
                self.embedding_dimension
            ),
        )

        self.vector_quantile_regression = VectorQuantileRegressor(
            solver=vector_quantile_regression_solver
        )

        self.vector_quantile_regression.fit(X_tensor, Y_tensor)
        self.fitted = True
        return self