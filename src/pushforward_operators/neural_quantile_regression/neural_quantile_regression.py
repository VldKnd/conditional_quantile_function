from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch
import torch.nn as nn
import time
from tqdm import trange
from typing import Literal
from utils.distribution import sample_distribution_like
from pushforward_operators.picnn import PISCNN


class NeuralQuantileRegression(PushForwardOperator, nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        potential_to_estimate_with_neural_network: Literal["y", "u"] = "u",
    ):
        super().__init__()

        self.init_dict = {
            "feature_dimension":
            feature_dimension,
            "response_dimension":
            response_dimension,
            "hidden_dimension":
            hidden_dimension,
            "number_of_hidden_layers":
            number_of_hidden_layers,
            "potential_to_estimate_with_neural_network":
            potential_to_estimate_with_neural_network
        }
        self.model_information_dict = {
            "class_name": "NeuralQuantileRegression",
        }

        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network
        self.potential_network = PISCNN(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
        )

        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

    def warmup_networks(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer_parameters: dict = {},
        warmup_iterations: int = 1,
        verbose: bool = False
    ):
        potential_network_optimizer = torch.optim.AdamW(
            self.potential_network.parameters(), **optimizer_parameters
        )

        progress_bar = trange(
            1, warmup_iterations + 1, desc="Warming up networks", disable=not verbose
        )
        for iteration in progress_bar:
            for X_batch, Y_batch in dataloader:
                Y_scaled = self.Y_scaler(Y_batch)
                U_batch = sample_distribution_like(Y_scaled, "normal")

                potential_network_optimizer.zero_grad()

                if self.potential_to_estimate_with_neural_network == "y":
                    Y_scaled.requires_grad_(True)
                    potential_tensor = self.potential_network(X_batch, Y_scaled)
                    potential_pushforward = torch.autograd.grad(
                        potential_tensor.sum(), Y_scaled, create_graph=True
                    )[0]
                    potential_loss = torch.norm(
                        potential_pushforward - Y_scaled, dim=-1
                    ).mean()
                    potential_loss.backward()
                else:
                    U_batch.requires_grad_(True)
                    potential_tensor = self.potential_network(X_batch, U_batch)
                    potential_pushforward = torch.autograd.grad(
                        potential_tensor.sum(), U_batch, create_graph=True
                    )[0]
                    potential_loss = torch.norm(
                        potential_pushforward - U_batch, dim=-1
                    ).mean()
                    potential_loss.backward()

                potential_network_optimizer.step()
                progress_bar.set_description(
                    f"Warm up iteration: {iteration} Potential loss: {potential_loss.item():.3f}"
                )

        return self

    def warmup_scalers(self, dataloader: torch.utils.data.DataLoader):
        """Run over the data (no grad) to populate BatchNorm running stats."""
        self.Y_scaler.train()

        with torch.no_grad():
            for _, Y in dataloader:
                _ = self.Y_scaler(Y)

        self.Y_scaler.eval()

    def make_progress_bar_message(
        self, training_information: list[dict], epoch_idx: int,
        last_learning_rate: float | None
    ):
        running_mean_objective = sum(
            [
                information["potential_loss"]
                for information in training_information[-10:]
            ]
        ) / len(training_information[-10:])

        return  (
            f"Epoch: {epoch_idx}, "
            f"Objective: {running_mean_objective:.3f}"
        ) + \
        (
            f", LR: {last_learning_rate[0]:.6f}"
            if last_learning_rate is not None
            else ""
        )

    @torch.enable_grad()
    def gradient_inverse(
        self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor
    ):
        point_tensor_clone = point_tensor.clone().requires_grad_(True)

        potential_value = self.potential_network(condition_tensor,
                                                 point_tensor_clone).sum()
        inverse_tensor = torch.autograd.grad(
            potential_value, point_tensor_clone, create_graph=False
        )[0]

        return inverse_tensor.detach()

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
        return inverse_tensor.detach()

    def fit(
        self, dataloader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters, *args, **kwargs
    ):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_parameters (TrainParameters): Training parameters.
        """
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        verbose = train_parameters.verbose
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)

        self.warmup_scalers(dataloader=dataloader)
        self.warmup_networks(
            dataloader=dataloader,
            optimizer_parameters=train_parameters.optimizer_parameters,
            warmup_iterations=train_parameters.warmup_iterations,
            verbose=verbose
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

        training_information = []
        training_information_per_epoch = []

        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        for epoch_idx in progress_bar:
            start_of_epoch = time.perf_counter()
            potential_losses_per_epoch = []

            for batch_index, (X_batch, Y_batch) in enumerate(dataloader):
                Y_scaled = self.Y_scaler(Y_batch)
                U_batch = sample_distribution_like(Y_batch, "normal")

                if self.potential_to_estimate_with_neural_network == "y":
                    inverse_tensor = self.c_transform_inverse(X_batch, U_batch)
                    Y_batch_for_phi, U_batch_for_psi = inverse_tensor, None
                else:
                    inverse_tensor = self.c_transform_inverse(X_batch, Y_scaled)
                    Y_batch_for_phi, U_batch_for_psi = None, inverse_tensor

                potential_network_optimizer.zero_grad()
                psi = self.estimate_psi(
                    X_tensor=X_batch, Y_tensor=Y_scaled, U_tensor=U_batch_for_psi
                )
                phi = self.estimate_phi(
                    X_tensor=X_batch, U_tensor=U_batch, Y_tensor=Y_batch_for_phi
                )
                potential_network_objective = torch.mean(phi) + torch.mean(psi)
                potential_network_objective.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.potential_network.parameters(), max_norm=1
                )
                potential_network_optimizer.step()

                if potential_network_scheduler is not None:
                    potential_network_scheduler.step()

                potential_losses_per_epoch.append(potential_network_objective.item())

                training_information.append(
                    {
                        "potential_loss":
                        potential_network_objective.item(),
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
                "potential_loss": torch.mean(torch.tensor(potential_losses_per_epoch)),
                "epoch_training_time": time.perf_counter() - start_of_epoch
            }
        )
        progress_bar.close()

        self.model_information_dict["number_of_epochs_to_train"
                                    ] = number_of_epochs_to_train
        self.model_information_dict["training_batch_size"] = dataloader.batch_size
        self.model_information_dict["training_information"
                                    ] = training_information_per_epoch
        return self

    def estimate_psi(
        self,
        X_tensor: torch.Tensor,
        Y_tensor: torch.Tensor,
        U_tensor: torch.Tensor | None = None
    ):
        """Estimates psi, either with Neural Network or by solving optimization with sgd.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "y":
            return self.potential_network(X_tensor, Y_tensor)
        else:
            return torch.sum(Y_tensor * U_tensor, dim=-1,
                             keepdim=True) - self.potential_network(X_tensor, U_tensor)

    def estimate_phi(
        self,
        X_tensor: torch.Tensor,
        U_tensor: torch.Tensor,
        Y_tensor: torch.Tensor | None = None
    ):
        """Estimates phi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "u":
            return self.potential_network(X_tensor, U_tensor)
        else:
            return torch.sum(Y_tensor * U_tensor, dim=-1,
                             keepdim=True) - self.potential_network(X_tensor, Y_tensor)

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        X_tensor = x
        Y_scaled = self.Y_scaler(y)

        if self.potential_to_estimate_with_neural_network == "y":
            U_tensor = self.gradient_inverse(X_tensor, Y_scaled)
        else:
            U_tensor = self.c_transform_inverse(X_tensor, Y_scaled)

        return U_tensor.detach()

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        X_tensor = x

        if self.potential_to_estimate_with_neural_network == "u":
            Y_tensor = self.gradient_inverse(X_tensor, u)
        else:
            Y_tensor = self.c_transform_inverse(X_tensor, u)

        return (
            Y_tensor.requires_grad_(False) * torch.sqrt(self.Y_scaler.running_var) +
            self.Y_scaler.running_mean
        ).detach()

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
    ) -> "NeuralQuantileRegression":
        data = torch.load(path, map_location=map_location)
        neural_quantile_regression = cls(**data["init_dict"])
        neural_quantile_regression.load_state_dict(data["state_dict"])
        neural_quantile_regression.model_information_dict = data.get(
            "model_information_dict", {}
        )
        return neural_quantile_regression
