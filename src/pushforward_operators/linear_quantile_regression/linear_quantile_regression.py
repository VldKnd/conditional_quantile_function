# Vector Quantile Regression, Guillaume Carlier et al https://arxiv.org/abs/1406.4643

import torch
import numpy
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr import VectorQuantileRegressor
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr.solvers.regularized_lse import RegularizedDualVQRSolver

_DEFAULT_SOLVER_ARGUMENTS = dict(
    T=50,
    epsilon=1e-3,
    num_epochs=1000,
    lr=0.9,
    lr_max_steps=10,
    lr_factor=0.5,
    lr_patience=100,
    lr_threshold=5 * 0.01,
    verbose=True,
    nn_init=None,
    batchsize_y=None,
    batchsize_u=None,
    inference_batch_size=1,
    full_precision=False,
    gpu=False,
    device_num=None,
    post_iter_callback=None,
)


class LinearQuantileRegression(PushForwardOperator, torch.nn.Module):

    def __init__(
        self,
        vector_quantile_regression_solver_arguments: dict = _DEFAULT_SOLVER_ARGUMENTS,
        *args,
        **kwargs
    ):
        super().__init__()
        self.vector_quantile_regression_arguments = vector_quantile_regression_solver_arguments
        self.vector_quantile_regression = None

    def fit_vector_quantile_regression(
        self, dataloader: torch.utils.data.DataLoader
    ) -> VectorQuantileRegressor:
        dataloader.shuffle = False
        Y_tensors = torch.cat([Y_tensor for _, Y_tensor in dataloader])
        X_tensors = torch.cat([X_tensor for X_tensor, _ in dataloader])
        nonlinear_mlp_solver = RegularizedDualVQRSolver(
            **self.vector_quantile_regression_arguments
        )
        vqr = VectorQuantileRegressor(solver=nonlinear_mlp_solver)
        return vqr.fit(X_tensors, Y_tensors)

    def make_progress_bar_message(
        self, training_information: list[dict], epoch_idx: int,
        last_learning_rate: float | None
    ):
        last_10_training_information = training_information[-10:]
        last_10_objectives = [
            information["potential_loss"]
            for information in last_10_training_information
        ]
        running_mean_objective = sum(last_10_objectives) / len(last_10_objectives)

        message = f"Epoch: {epoch_idx}, Objective: {running_mean_objective:.3f}"
        if last_learning_rate is not None:
            message += f", LR: {last_learning_rate[0]:.6f}"

        return message

    def fit(self, dataloader: torch.utils.data.DataLoader, *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
            verbose (bool): Whether to print verbose output.
        """
        self.vector_quantile_regression = self.fit_vector_quantile_regression(
            dataloader=dataloader
        )

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-1]
        y_dimension = y.shape[-1]

        x_flat = x.flatten(start_dim=0, end_dim=-2)
        y_flat = y.flatten(start_dim=0, end_dim=-2)
        u_samples = []

        for x_single, y_single in zip(x_flat, y_flat):
            x_numpy = x_single.numpy(force=True)
            y_numpy = y_single.numpy(force=True)
            vector_quantile_function = self.vector_quantile_regression.vector_quantiles(
                X=x_numpy
            )[0]
            y_quantile_surfaces = numpy.stack(tuple(vector_quantile_function))
            y_quantile_surfaces_flat = y_quantile_surfaces.reshape(y_dimension, -1)
            u_quantile_surfaces_flat = vector_quantile_function.quantile_grid.reshape(
                y_dimension, -1
            )
            distances_to_quantile_surfaces = numpy.linalg.norm(
                y_quantile_surfaces_flat - y_numpy.reshape(-1, 1), axis=0
            )
            index_of_closest_quantile_vector = numpy.argmin(
                distances_to_quantile_surfaces
            )
            u_samples.append(
                torch.tensor(
                    u_quantile_surfaces_flat[:, index_of_closest_quantile_vector]
                )
            )

        u_flat = torch.stack(u_samples).to(x)
        return u_flat.reshape(*input_shape, -1)

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-1]
        u_dimension = u.shape[-1]

        x_flat = x.flatten(start_dim=0, end_dim=-2)
        u_flat = u.flatten(start_dim=0, end_dim=-2)
        y_samples = []

        for x_single, u_single in zip(x_flat, u_flat):
            x_numpy = x_single.numpy(force=True)
            u_numpy = u_single.numpy(force=True)
            vector_quantile_function = self.vector_quantile_regression.vector_quantiles(
                X=x_numpy
            )[0]
            y_quantile_surfaces = numpy.stack(tuple(vector_quantile_function))
            y_quantile_surfaces_flat = y_quantile_surfaces.reshape(u_dimension, -1)
            u_quantile_surfaces_flat = vector_quantile_function.quantile_grid.reshape(
                u_dimension, -1
            )
            distances_to_quantile_surfaces = numpy.linalg.norm(
                u_quantile_surfaces_flat - u_numpy.reshape(-1, 1), axis=0
            )
            index_of_closest_quantile_vector = numpy.argmin(
                distances_to_quantile_surfaces
            )
            y_samples.append(
                torch.tensor(
                    y_quantile_surfaces_flat[:, index_of_closest_quantile_vector]
                )
            )

        y_flat = torch.stack(y_samples).to(x)
        return y_flat.reshape(*input_shape, -1)

    def save(self, path: str):
        import pickle
        file = open(path, 'wb')
        pickle.dump(self.vector_quantile_regression, file)
        file.close()

    def load(self, path: str, *args, **kwargs):
        import pickle
        file = open(path, 'rb')
        self.vector_quantile_regression = pickle.load(file)
        file.close()

    @classmethod
    def load_class(cls, path: str, *args, **kwargs) -> "LinearQuantileRegression":
        linear_quantile_regression = cls()
        linear_quantile_regression.load(path, *args, **kwargs)
        return linear_quantile_regression
