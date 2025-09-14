import torch
from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
from tqdm import trange
import time
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr import VectorQuantileRegressor
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr.solvers.regularized_lse import MLPRegularizedDualVQRSolver
from pushforward_operators.picnn import PISCNN


class FNLVQR(PushForwardOperator, torch.nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        fnlvqr_mlp_arguments: dict = {},
        *args,
        **kwargs
    ):
        super().__init__()
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "fnlvqr_mlp_arguments": fnlvqr_mlp_arguments,
        }
        self.model_information_dict = {
            "name": "Fast Non Linear Vector Quantile Regression"
        }
        self.fnlvqr_mlp_arguments = fnlvqr_mlp_arguments
        self.fnlvqr = None

        self.potential_network = PISCNN(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
        )

    def fit_fnlvqr(
        self, dataloader: torch.utils.data.DataLoader
    ) -> VectorQuantileRegressor:
        dataloader.shuffle = False
        Y_tensors = torch.cat([Y_tensor for _, Y_tensor in dataloader])
        X_tensors = torch.cat([X_tensor for X_tensor, _ in dataloader])
        nonlinear_mlp_solver = MLPRegularizedDualVQRSolver(**self.fnlvqr_mlp_arguments)
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

    def fit(
        self, dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters, *args, **kwargs
    ):
        """Fits the pushforward operator to the data.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
            verbose (bool): Whether to print verbose output.
        """
        self.fnlvqr = self.fit_fnlvqr(dataloader=dataloader)
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        verbose = train_parameters.verbose

        training_information = []
        training_information_per_epoch = []
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)

        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        potential_network_optimizer = torch.optim.AdamW(
            self.potential_network.parameters(), **train_parameters.optimizer_parameters
        )

        if train_parameters.scheduler_parameters:
            potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                potential_network_optimizer, total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            potential_network_scheduler = None

        for epoch_idx in progress_bar:
            start_of_epoch = time.perf_counter()
            potential_losses_per_epoch = []

            for batch_index, (X_batch, _) in enumerate(dataloader):
                y_samples = []
                u_samples = []

                for x in X_batch:
                    y, u = self.fnlvqr.sample_one_conditional_element(x)
                    y_samples.append(y)
                    u_samples.append(u)

                Y_batch = torch.cat(y_samples).to(X_batch)
                U_batch = torch.cat(u_samples).to(X_batch)
                U_batch.requires_grad_(True)

                potential_network_optimizer.zero_grad()
                U_potential = self.potential_network(X_batch, U_batch)
                Y_pushforward = torch.autograd.grad(
                    U_potential.sum(), U_batch, create_graph=True
                )[0]

                potential_objective = Y_batch.sub(Y_pushforward).norm(dim=-1).mean()
                potential_objective.backward()

                potential_network_optimizer.step()

                if potential_network_scheduler is not None:
                    potential_network_scheduler.step()

                potential_losses_per_epoch.append(potential_objective.item())

                training_information.append(
                    {
                        "potential_loss":
                        potential_objective.item(),
                        "batch_index":
                        batch_index,
                        "epoch_index":
                        epoch_idx,
                        "time_elapsed_since_last_epoch":
                        time.perf_counter() - start_of_epoch,
                    }
                )

                if verbose:
                    last_learning_rate = (
                        potential_network_scheduler.get_last_lr()
                        if potential_network_scheduler is not None else None
                    )

                    progress_bar_message = self.make_progress_bar_message(
                        training_information=training_information,
                        epoch_idx=epoch_idx,
                        last_learning_rate=last_learning_rate
                    )

                    progress_bar.set_description(progress_bar_message)

            training_information_per_epoch.append(
                {
                    "potential_loss":
                    torch.mean(torch.tensor(potential_losses_per_epoch)),
                    "epoch_training_time": time.perf_counter() - start_of_epoch
                }
            )

        progress_bar.close()

        self.model_information_dict["number_of_epochs_to_train"
                                    ] = number_of_epochs_to_train
        self.model_information_dict["training_batch_size"] = dataloader.batch_size
        self.model_information_dict['training_information'
                                    ] = training_information_per_epoch

        return self

    def c_transform_inverse(
        self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor
    ):
        inverse_tensor = torch.nn.Parameter(torch.randn_like(point_tensor).contiguous())

        optimizer = torch.optim.LBFGS(
            [inverse_tensor],
            lr=1,
            line_search_fn="strong_wolfe",
            max_iter=1000,
            tolerance_grad=1e-7,
            tolerance_change=1e-7
        )

        def slackness_closure():
            optimizer.zero_grad()
            cost_matrix = torch.sum(point_tensor * inverse_tensor, dim=-1, keepdim=True)
            potential = self.potential_network(condition_tensor, inverse_tensor)
            slackness = (potential - cost_matrix).mean()
            slackness.backward()
            return slackness

        optimizer.step(slackness_closure)
        inverse_tensor = inverse_tensor.detach()
        inverse_tensor = inverse_tensor.less_equal(1).mul(
            inverse_tensor
        ) + inverse_tensor.greater(1).mul(1)
        inverse_tensor = inverse_tensor.greater_equal(0).mul(
            inverse_tensor
        ) + inverse_tensor.less(0).mul(0)
        return inverse_tensor.detach()

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.c_transform_inverse(
            condition_tensor=x,
            point_tensor=y,
        )

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        X_tensor, U_tensor = x, u.clone().requires_grad_(True)
        Y_tensor = torch.autograd.grad(
            self.potential_network(X_tensor, U_tensor).sum(),
            U_tensor,
            create_graph=False
        )[0]
        return Y_tensor.detach()

    def save(self, path: str):
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "model_information_dict": self.model_information_dict,
            }, path
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self

    @classmethod
    def load_class(
        cls, path: str, map_location: torch.device = torch.device('cpu')
    ) -> "FNLVQR":
        data = torch.load(path, map_location=map_location)
        fnlvqr = cls(**data["init_dict"])
        fnlvqr.load_state_dict(data["state_dict"])
        fnlvqr.model_information_dict = data.get("model_information_dict", {})
        return fnlvqr
