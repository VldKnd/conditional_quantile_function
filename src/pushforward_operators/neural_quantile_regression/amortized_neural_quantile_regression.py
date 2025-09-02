from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch
import torch.nn as nn
from typing import Literal
from pushforward_operators.picnn import network_type_name_to_network_type

from tqdm import trange


class AmortizationNetwork(nn.Module):

    def __init__(
        self, feature_dimension: int, response_dimension: int, hidden_dimension: int,
        number_of_hidden_layers: int
    ):
        super().__init__()
        self.activation_function = nn.ReLU()
        self.feature_expansion_layer = nn.Linear(
            feature_dimension, response_dimension * 2
        )

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.amortization_network = nn.Sequential(
            nn.Linear(3 * response_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            nn.Linear(hidden_dimension, response_dimension)
        )

        self.identity_projection = nn.Linear(response_dimension, response_dimension)

    def forward(self, X: torch.Tensor, U: torch.Tensor):
        input_tensor = torch.cat([self.feature_expansion_layer(X), U], dim=-1)
        output_tensor = self.amortization_network(input_tensor)
        input_projection = self.identity_projection(U)
        return output_tensor + input_projection


class AmortizedNeuralQuantileRegression(PushForwardOperator, nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        network_type: Literal["SCFFNN", "PISCNN"] = "PISCNN",
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
            "network_type":
            network_type,
            "potential_to_estimate_with_neural_network":
            potential_to_estimate_with_neural_network
        }
        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network

        self.potential_network = network_type_name_to_network_type[network_type](
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
        )

        self.amortization_network = AmortizationNetwork(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
        )

        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

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
        running_mean_potential_objective = sum(
            [information["potential"] for information in training_information[-10:]]
        ) / len(training_information[-10:])
        running_mean_amortization_objective = sum(
            [information["amortization"] for information in training_information[-10:]]
        ) / len(training_information[-10:])

        description_message = (
            f"Epoch: {epoch_idx}, "
            f"Potential Objective: {running_mean_potential_objective:.3f}, "
            f"Amortization Objective: {running_mean_amortization_objective:.3f}"
        ) + (
            f", LR: {last_learning_rate:.6f}" if last_learning_rate is not None else ""
        )

        return description_message

    def gradient_inverse(
        self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor
    ):
        requires_grad_backup, point_tensor.requires_grad = point_tensor.requires_grad, True
        inverse_tensor = torch.autograd.grad(
            self.potential_network(condition_tensor, point_tensor).sum(),
            point_tensor,
            create_graph=False
        )[0]
        point_tensor.requires_grad = requires_grad_backup
        return inverse_tensor.detach()

    def c_transform_inverse(
        self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor,
        approximation_of_inverse_tensor: torch.Tensor
    ):
        inverse_tensor = torch.nn.Parameter(
            approximation_of_inverse_tensor.clone().contiguous()
        )

        optimizer = torch.optim.LBFGS(
            [inverse_tensor],
            lr=1,
            line_search_fn="strong_wolfe",
            max_iter=1000,
            tolerance_grad=1e-10,
            tolerance_change=1e-10
        )

        def slackness_closure():
            optimizer.zero_grad()
            cost_matrix = torch.sum(point_tensor * inverse_tensor, dim=-1, keepdim=True)
            potential = self.potential_network(condition_tensor, inverse_tensor)
            slackness = (potential - cost_matrix).sum()
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
        potential_network_optimizer = torch.optim.AdamW(
            params=self.potential_network.parameters(),
            **train_parameters.optimizer_parameters
        )
        amortization_network_optimizer = torch.optim.AdamW(
            self.amortization_network.parameters(),
            **train_parameters.optimizer_parameters
        )

        if train_parameters.scheduler_parameters:
            potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=potential_network_optimizer,
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
            amortization_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=potential_network_optimizer,
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            amortization_network_scheduler = None
            potential_network_scheduler = None

        training_information = []
        self.warmup_scalers(dataloader=dataloader)
        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        for epoch_idx in progress_bar:
            for X_batch, Y_batch in dataloader:
                Y_scaled = self.Y_scaler(Y_batch)
                U_batch = torch.randn_like(Y_batch)

                if self.potential_to_estimate_with_neural_network == "y":
                    amortized_tensor = self.amortization_network(X_batch, U_batch)
                    inverse_tensor = self.c_transform_inverse(
                        X_batch, U_batch, amortized_tensor
                    )
                    Y_batch_for_phi, U_batch_for_psi = inverse_tensor, None
                else:
                    amortized_tensor = self.amortization_network(X_batch, Y_scaled)
                    inverse_tensor = self.c_transform_inverse(
                        X_batch, Y_scaled, amortized_tensor
                    )
                    Y_batch_for_phi, U_batch_for_psi = None, inverse_tensor

                self.amortization_network.zero_grad()
                amortization_network_objective = torch.norm(
                    amortized_tensor - inverse_tensor, dim=-1
                ).mean()
                amortization_network_objective.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.amortization_network.parameters(), max_norm=10
                )
                amortization_network_optimizer.step()

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
                    self.potential_network.parameters(), max_norm=10
                )
                potential_network_optimizer.step()

                if potential_network_scheduler is not None and amortization_network_scheduler is not None:
                    potential_network_scheduler.step()
                    amortization_network_scheduler.step()

                if verbose:
                    training_information.append(
                        {
                            "potential": potential_network_objective.item(),
                            "amortization": amortization_network_objective.item(),
                            "epoch_index": epoch_idx
                        }
                    )

                    last_learning_rate = (
                        amortization_network_scheduler.get_last_lr()[0]
                        if amortization_network_scheduler is not None else None
                    )

                    description_message = self.make_progress_bar_message(
                        training_information=training_information,
                        epoch_idx=epoch_idx,
                        last_learning_rate=last_learning_rate
                    )

                    progress_bar.set_description(description_message)

        progress_bar.close()
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
    def push_y_given_x(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        u_initial: torch.Tensor | None = None
    ) -> torch.Tensor:
        X_tensor = x
        Y_scaled = self.Y_scaler(y)

        if self.potential_to_estimate_with_neural_network == "y":
            U_tensor = self.gradient_inverse(X_tensor, Y_scaled)
        else:
            if u_initial is None:
                u_initial = self.amortization_network(X_tensor, Y_scaled)
            U_tensor = self.c_transform_inverse(X_tensor, Y_scaled, u_initial)

        return U_tensor.requires_grad_(False).detach()

    @torch.enable_grad()
    def push_u_given_x(
        self,
        u: torch.Tensor,
        x: torch.Tensor,
        y_initial: torch.Tensor | None = None
    ) -> torch.Tensor:
        X_tensor = x

        if self.potential_to_estimate_with_neural_network == "u":
            Y_tensor = self.gradient_inverse(X_tensor, u)
        else:
            if y_initial is None:
                y_initial = self.amortization_network(X_tensor, u)
            Y_tensor = self.c_transform_inverse(X_tensor, u, y_initial)

        return (
            Y_tensor.requires_grad_(False) * torch.sqrt(self.Y_scaler.running_var) +
            self.Y_scaler.running_mean
        ).detach()

    def save(self, path: str):
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "class_name": "AmortizedNeuralQuantileRegression"
            }, path
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self

    @classmethod
    def load_class(
        cls, path: str, map_location: torch.device = torch.device('cpu')
    ) -> "AmortizedNeuralQuantileRegression":
        data = torch.load(path, map_location=map_location)
        operator = cls(**data["init_dict"])
        operator.load_state_dict(data["state_dict"])
        return operator
