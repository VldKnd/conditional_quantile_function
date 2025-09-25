from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch
import time
import torch.nn as nn
from typing import Literal
from pushforward_operators.picnn import PISCNN

from tqdm import trange
from utils.distribution import sample_distribution_like, sample_uniform_ball_surface, sample_uniform_ball_surface_like


class AmortizationNetwork(nn.Module):

    def __init__(
        self, feature_dimension: int, response_dimension: int, hidden_dimension: int,
        number_of_hidden_layers: int
    ):
        super().__init__()
        self.activation_function = nn.ReLU()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.amortization_network = nn.Sequential(
            nn.Linear(feature_dimension + response_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            nn.Linear(hidden_dimension, response_dimension)
        )

        self.identity_projection = nn.Linear(response_dimension, response_dimension)

    def forward(self, X: torch.Tensor, U: torch.Tensor):
        input_tensor = torch.cat([X, U], dim=-1)
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
            potential_to_estimate_with_neural_network,
        }
        self.response_dimension = response_dimension
        self.model_information_dict = {
            "class_name": "AmortizedNeuralQuantileRegression",
        }
        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network

        self.potential_network = PISCNN(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers
        )

        self.amortization_network = AmortizationNetwork(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
        )

        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

    def get_log_volume(
            self,
            condition: torch.Tensor,
            radius: float,
            number_of_points_to_estimate_bounding_box: int = 1000,
            number_of_points_to_estimate_volume: int = 1000,
        ) -> float:
        """
        Computes volume of a region created by pushing sphere of radius `radius` from U space to Y by applying push_u_given_x.
        """
        condition_squeezed = condition.squeeze()
        if len(condition_squeezed.shape) > 1:
            raise RuntimeError("Condition should have only one value in form [..., dimension_of_x]")
        
        condition_expanded_for_ball_in_u = condition_squeezed.unsqueeze(0).repeat(number_of_points_to_estimate_bounding_box, 1)
        u_dimension = self.response_dimension
        ball_in_u_shape = (number_of_points_to_estimate_bounding_box, u_dimension)
        ball_in_u = radius * sample_uniform_ball_surface(ball_in_u_shape)
        ball_in_u = ball_in_u.to(condition)

        transformed_ball_surface_in_y = self.push_u_given_x(u=ball_in_u, x=condition_expanded_for_ball_in_u)
        bounding_box_in_y_max, _ = transformed_ball_surface_in_y.max(dim=0)
        bounding_box_in_y_min, _ = transformed_ball_surface_in_y.min(dim=0)

        samples_from_bounding_box_interiour_in_y = torch.rand(number_of_points_to_estimate_volume, u_dimension)
        samples_from_bounding_box_interiour_in_y = samples_from_bounding_box_interiour_in_y.to(condition) * (bounding_box_in_y_max - bounding_box_in_y_min) + bounding_box_in_y_min
        condition_expanded_for_bounding_box_in_y = condition_squeezed.unsqueeze(0).repeat(number_of_points_to_estimate_volume, 1)
        bounding_box_inferiour_in_u = self.push_y_given_x(y=samples_from_bounding_box_interiour_in_y, x=condition_expanded_for_bounding_box_in_y)
        
        log_percentage_of_points_in_the_ball_in_u = bounding_box_inferiour_in_u.norm(dim=-1).less_equal(radius).float().mean().add(1e-15).log()
        log_volume_of_bounding_box_in_y = (bounding_box_in_y_max - bounding_box_in_y_min).log().sum()

        return log_volume_of_bounding_box_in_y + log_percentage_of_points_in_the_ball_in_u

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
        amortization_network_optimizer = torch.optim.AdamW(
            self.amortization_network.parameters(), **optimizer_parameters
        )

        progress_bar = trange(
            1, warmup_iterations + 1, desc="Warming up networks", disable=not verbose
        )
        for iteration in progress_bar:
            for X_batch, Y_batch in dataloader:
                Y_scaled = self.Y_scaler(Y_batch)
                U_batch = sample_distribution_like(Y_scaled, "normal")

                amortization_network_optimizer.zero_grad()
                potential_network_optimizer.zero_grad()

                if self.potential_to_estimate_with_neural_network == "y":
                    amortized_tensor = self.amortization_network(X_batch, U_batch)
                    amortized_loss = torch.norm(amortized_tensor - U_batch,
                                                dim=-1).mean()
                    amortized_loss.backward()

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
                    amortized_tensor = self.amortization_network(X_batch, Y_scaled)
                    amortized_loss = torch.norm(amortized_tensor - Y_scaled,
                                                dim=-1).mean()
                    amortized_loss.backward()

                    U_batch.requires_grad_(True)
                    potential_tensor = self.potential_network(X_batch, U_batch)
                    potential_pushforward = torch.autograd.grad(
                        potential_tensor.sum(), U_batch, create_graph=True
                    )[0]
                    potential_loss = torch.norm(
                        potential_pushforward - U_batch, dim=-1
                    ).mean()
                    potential_loss.backward()

                amortization_network_optimizer.step()
                potential_network_optimizer.step()
                progress_bar.set_description(
                    f"Warm up iteration: {iteration} Potential loss: {potential_loss.item():.3f}, Amortization loss: {amortized_loss.item():.3f}"
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
        self,
        training_information: list[dict],
        epoch_idx: int,
        last_amortization_network_learning_rate: float | None = None,
        last_potential_network_learning_rate: float | None = None
    ):
        running_mean_potential_objective = sum(
            [
                information["potential_loss"]
                for information in training_information[-10:]
            ]
        ) / len(training_information[-10:])
        running_mean_amortization_objective = sum(
            [
                information["amortization_loss"]
                for information in training_information[-10:]
            ]
        ) / len(training_information[-10:])

        description_message = (
            f"Epoch: {epoch_idx}, "
            f"Potential Objective: {running_mean_potential_objective:.3f}, "
            f"Amortization Objective: {running_mean_amortization_objective:.3f}"
        ) + (
            f", Potential LR: {last_potential_network_learning_rate:.6f}"
            if last_potential_network_learning_rate is not None else ""
        ) + (
            f", Amortized LR: {last_amortization_network_learning_rate:.6f}"
            if last_amortization_network_learning_rate is not None else ""
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

        self.warmup_scalers(dataloader=dataloader)
        self.warmup_networks(
            dataloader=dataloader,
            optimizer_parameters=train_parameters.optimizer_parameters,
            warmup_iterations=train_parameters.warmup_iterations,
            verbose=verbose
        )

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

            amortization_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=amortization_network_optimizer,
                T_0=max((total_number_of_optimizer_steps) // 100, 1),
                **train_parameters.scheduler_parameters
            )

        else:
            amortization_network_scheduler = None
            potential_network_scheduler = None

        training_information = []
        training_information_per_epoch = []

        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        for epoch_idx in progress_bar:
            start_of_epoch = time.perf_counter()
            amortization_losses_per_epoch = []
            potential_losses_per_epoch = []

            for batch_index, (X_batch, Y_batch) in enumerate(dataloader):
                Y_scaled = self.Y_scaler(Y_batch)
                U_batch = sample_distribution_like(Y_batch, "normal")

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

                amortization_network_optimizer.zero_grad()
                amortization_network_objective = torch.norm(
                    amortized_tensor - inverse_tensor, dim=-1
                ).mean()
                amortization_network_objective.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.amortization_network.parameters(), max_norm=1
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
                    self.potential_network.parameters(), max_norm=1.
                )
                potential_network_optimizer.step()

                if potential_network_scheduler is not None and amortization_network_scheduler is not None:
                    potential_network_scheduler.step()
                    amortization_network_scheduler.step()

                amortization_losses_per_epoch.append(
                    amortization_network_objective.item()
                )
                potential_losses_per_epoch.append(potential_network_objective.item())

                training_information.append(
                    {
                        "potential_loss": potential_network_objective.item(),
                        "amortization_loss": amortization_network_objective.item(),
                        "batch_index": batch_index,
                        "epoch_index": epoch_idx,
                    }
                )

                if verbose:
                    last_amortization_network_learning_rate = (
                        amortization_network_scheduler.get_last_lr()[0]
                        if amortization_network_scheduler is not None else None
                    )
                    last_potential_network_learning_rate = (
                        potential_network_scheduler.get_last_lr()[0]
                        if potential_network_scheduler is not None else None
                    )

                    description_message = self.make_progress_bar_message(
                        training_information=training_information,
                        epoch_idx=epoch_idx,
                        last_amortization_network_learning_rate=
                        last_amortization_network_learning_rate,
                        last_potential_network_learning_rate=
                        last_potential_network_learning_rate,
                    )

                    progress_bar.set_description(description_message)

            training_information_per_epoch.append(
                {
                    "potential_loss":
                    torch.mean(torch.tensor(potential_losses_per_epoch)),
                    "amortization_loss":
                    torch.mean(torch.tensor(amortization_losses_per_epoch)),
                    "epoch_training_time":
                    time.perf_counter() - start_of_epoch
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
        Y_tensor: torch.Tensor | None = None,
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
    ) -> "AmortizedNeuralQuantileRegression":
        data = torch.load(path, map_location=map_location)
        amortized_neural_quantile_regression = cls(**data["init_dict"])
        amortized_neural_quantile_regression.load_state_dict(data["state_dict"])
        amortized_neural_quantile_regression.model_information_dict = data.get(
            "model_information_dict", {}
        )
        return amortized_neural_quantile_regression
