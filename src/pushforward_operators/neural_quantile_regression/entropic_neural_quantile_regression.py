import torch.nn as nn
import torch
import time
from tqdm import trange
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators.picnn import PICNN
from utils.distribution import sample_distribution_like, sample_distribution, sample_uniform_ball_surface


class EntropicNeuralQuantileRegression(PushForwardOperator, nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        epsilon: float,
        amount_of_samples_to_estimate_psi: int = 1024,
        *args,
        **kwargs
    ):
        super().__init__()
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "amount_of_samples_to_estimate_psi": amount_of_samples_to_estimate_psi,
            "epsilon": epsilon,
        }
        self.model_information_dict = {
            "class_name": "EntropicNeuralQuantileRegression",
        }

        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

        self.potential_network = PICNN(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers
        )

        self.epsilon = epsilon
        self.amount_of_samples_to_estimate_psi = amount_of_samples_to_estimate_psi

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
            raise RuntimeError(
                "Condition should have only one value in form [..., dimension_of_x]"
            )

        condition_expanded_for_ball_in_u = condition_squeezed.unsqueeze(0).repeat(
            number_of_points_to_estimate_bounding_box, 1
        )
        u_dimension = self.response_dimension
        ball_in_u_shape = (number_of_points_to_estimate_bounding_box, u_dimension)
        ball_in_u = radius * sample_uniform_ball_surface(ball_in_u_shape)
        ball_in_u = ball_in_u.to(condition)

        transformed_ball_surface_in_y = self.push_u_given_x(
            u=ball_in_u, x=condition_expanded_for_ball_in_u
        )
        bounding_box_in_y_max, _ = transformed_ball_surface_in_y.max(dim=0)
        bounding_box_in_y_min, _ = transformed_ball_surface_in_y.min(dim=0)

        samples_from_bounding_box_interiour_in_y = torch.rand(
            number_of_points_to_estimate_volume, u_dimension
        )
        samples_from_bounding_box_interiour_in_y = samples_from_bounding_box_interiour_in_y.to(
            condition
        ) * (bounding_box_in_y_max - bounding_box_in_y_min) + bounding_box_in_y_min
        condition_expanded_for_bounding_box_in_y = condition_squeezed.unsqueeze(
            0
        ).repeat(number_of_points_to_estimate_volume, 1)
        bounding_box_inferiour_in_u = self.push_y_given_x(
            y=samples_from_bounding_box_interiour_in_y,
            x=condition_expanded_for_bounding_box_in_y
        )

        log_percentage_of_points_in_the_ball_in_u = bounding_box_inferiour_in_u.norm(
            dim=-1
        ).less_equal(radius).float().mean().add(1e-15).log()
        log_volume_of_bounding_box_in_y = (
            bounding_box_in_y_max - bounding_box_in_y_min
        ).log().sum()

        return log_volume_of_bounding_box_in_y + log_percentage_of_points_in_the_ball_in_u

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

                U_batch.requires_grad_(True)
                potential_tensor = self.potential_network(X_batch, U_batch)
                potential_pushforward = torch.autograd.grad(
                    potential_tensor.sum(), U_batch, create_graph=True
                )[0]
                potential_loss = torch.norm(potential_pushforward - U_batch,
                                            dim=-1).mean()
                potential_loss.backward()

                potential_network_optimizer.step()

                progress_bar.set_description(
                    f"Warm up iteration: {iteration} Potential loss: {potential_loss.item():.3f}"
                )

        return self

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

        if train_parameters.scheduler_parameters:
            potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=potential_network_optimizer,
                T_max=total_number_of_optimizer_steps,
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

                potential_network_optimizer.zero_grad()
                psi = self.estimate_psi(X_tensor=X_batch, Y_tensor=Y_scaled)
                phi = self.estimate_phi(X_tensor=X_batch, U_tensor=U_batch)
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

    def estimate_psi(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates psi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            Y_tensor (torch.Tensor): Output tensor.
        """
        n, _ = X_tensor.shape
        m = self.amount_of_samples_to_estimate_psi
        U_tensor = sample_distribution(
            (self.amount_of_samples_to_estimate_psi, *Y_tensor.shape[1:]), "normal"
        ).to(Y_tensor)
        U_expanded_for_X = U_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_U = X_tensor.unsqueeze(1).expand(-1, m, -1)

        phi_values = self.potential_network(
            X_expanded_for_U.reshape(-1, X_expanded_for_U.shape[-1]),
            U_expanded_for_X.reshape(-1, U_expanded_for_X.shape[-1])
        ).reshape(n, m, -1).squeeze(-1)

        cost_matrix = Y_tensor @ U_tensor.T

        slackness = (cost_matrix - phi_values) / self.epsilon
        log_mean_exp = torch.logsumexp(slackness, dim=-1, keepdim=True) \
                - torch.log(torch.tensor(m, device=slackness.device, dtype=slackness.dtype))

        psi_estimate = self.epsilon * log_mean_exp

        return psi_estimate

    def estimate_phi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor):
        """Estimates phi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        return self.potential_network(X_tensor, U_tensor)

    def gradient_inverse(
        self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor
    ):
        inverse_tensor = torch.autograd.grad(
            self.potential_network(condition_tensor, point_tensor).sum(),
            point_tensor,
            create_graph=False
        )[0]
        return inverse_tensor.detach()

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        X_tensor = x
        Y_scaled = self.Y_scaler(y).requires_grad_(True)

        psi_potential = self.estimate_psi(X_tensor=X_tensor, Y_tensor=Y_scaled)
        Y_pushforward = torch.autograd.grad(
            psi_potential.sum(), Y_scaled, create_graph=False
        )[0]
        return Y_pushforward.detach()

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        X_tensor, U_tensor = x, u.clone().requires_grad_(True)
        Y_tensor = self.gradient_inverse(
            condition_tensor=X_tensor, point_tensor=U_tensor
        )

        return (
            Y_tensor * torch.sqrt(self.Y_scaler.running_var) +
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
    ) -> "EntropicNeuralQuantileRegression":
        data = torch.load(path, map_location=map_location)
        entropic_neural_quantile_regression = cls(**data["init_dict"])
        entropic_neural_quantile_regression.load_state_dict(data["state_dict"])
        entropic_neural_quantile_regression.model_information_dict = data.get(
            "model_information_dict", {}
        )
        return entropic_neural_quantile_regression
