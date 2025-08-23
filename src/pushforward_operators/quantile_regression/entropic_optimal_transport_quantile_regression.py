import torch.nn as nn
import torch
from tqdm import trange
from typing import Literal
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators.picnn import network_type_name_to_network_type

class EntropicOTQuantileRegression(PushForwardOperator, nn.Module):
    def __init__(self,
            feature_dimension: int,
            response_dimension: int,
            hidden_dimension: int,
            number_of_hidden_layers: int,
            epsilon: float,
            activation_function_name: str = "Softplus",
            network_type: Literal["SCFFNN", "FFNN", "PICNN", "PISCNN"] = "FFNN",
            potential_to_estimate_with_neural_network: Literal["y", "u"] = "y",
        ):
        super().__init__()
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "epsilon":epsilon,
            "activation_function_name": activation_function_name,
            "network_type":network_type,
            "potential_to_estimate_with_neural_network":potential_to_estimate_with_neural_network
        }

        self.activation_function_name = activation_function_name
        self.activation_function = getattr(nn, activation_function_name)()
        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)
        self.X_scaler = nn.BatchNorm1d(feature_dimension, affine=False)

        self.network_type = network_type
        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network 

        self.potential_network = network_type_name_to_network_type[network_type](
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name
        )

        self.epsilon = epsilon
    
    def warmup_scalers(self, dataloader: torch.utils.data.DataLoader):
        """Run over the data (no grad) to populate BatchNorm running stats."""
        self.X_scaler.train(), self.Y_scaler.train()

        with torch.no_grad():
            for X, Y in dataloader:
                _, _ = self.Y_scaler(Y), self.X_scaler(X)

        self.X_scaler.eval(), self.Y_scaler.eval()

    def make_progress_bar_message(self, training_information: list[dict], epoch_idx:int, last_learning_rate: float | None):
        running_mean_objective = sum([information["objective"] for information in training_information[-10:]]) / len(training_information[-10:])
                        
        return  (
            f"Epoch: {epoch_idx}, "
            f"Objective: {running_mean_objective:.3f}"
        ) + \
        (
            f", LR: {last_learning_rate[0]:.6f}"
            if last_learning_rate is not None
            else ""
        )


    def fit(self, dataloader: torch.utils.data.DataLoader, train_parameters: TrainParameters, *args, **kwargs):
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
        if train_parameters.scheduler_parameters:
            potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=potential_network_optimizer, 
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            potential_network_scheduler = None

        training_information = []
        self.warmup_scalers(dataloader=dataloader)
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)
        
        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    self.potential_network.zero_grad()

                    X_scaled = self.X_scaler(X_batch)
                    Y_scaled = self.Y_scaler(Y_batch)
                    U_batch = torch.randn_like(Y_scaled)


                    potential_network_optimizer.zero_grad()
                    psi = self.estimate_psi(X_tensor=X_scaled, Y_tensor=Y_scaled, U_tensor=U_batch)
                    phi = self.estimate_phi(X_tensor=X_scaled, Y_tensor=Y_scaled, U_tensor=U_batch)
                    potential_network_objective = torch.mean(phi) + torch.mean(psi)
                    potential_network_objective.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(self.potential_network.parameters(), max_norm=10)
                    potential_network_optimizer.step()
                    
                    if potential_network_scheduler is not None:
                        potential_network_scheduler.step()

                    if verbose:

                        training_information.append({
                                "objective": potential_network_objective.item(),
                                "epoch_index": epoch_idx
                        })

                        last_learning_rate = (
                            potential_network_scheduler.get_last_lr() 
                            if potential_network_scheduler is not None
                            else None
                        )
                        
                        progress_bar_message = self.make_progress_bar_message(
                            training_information=training_information,
                            epoch_idx=epoch_idx,
                            last_learning_rate=last_learning_rate
                        )

                        progress_bar.set_description(progress_bar_message)

        progress_bar.close()
        return self
    
    def estimate_psi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates psi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "y":
            return self.potential_network(X_tensor, Y_tensor)
        
        n, _ = X_tensor.shape

        U_expanded_for_X = U_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_U = X_tensor.unsqueeze(1).expand(-1, n, -1)

        phi_values = self.potential_network(X_expanded_for_U, U_expanded_for_X).squeeze(-1)
        cost_matrix = Y_tensor @ U_tensor.T

        slackness = cost_matrix - phi_values
        max_slackness, _ = torch.max(slackness, dim=1, keepdim=True)
        slackness_stable = (slackness - max_slackness) / self.epsilon
        log_mean_exp = torch.logsumexp(slackness_stable, dim=1, keepdim=True) \
                - torch.log(torch.tensor(n, device=slackness.device, dtype=slackness.dtype))
        
        log_mean_exp += max_slackness / self.epsilon

        phi_estimate = self.epsilon * log_mean_exp

        return phi_estimate
    
    def estimate_phi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates phi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "u":
            return self.potential_network(X_tensor, U_tensor)
        
        n, _ = X_tensor.shape

        Y_expanded_for_X = Y_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_Y = X_tensor.unsqueeze(1).expand(-1, n, -1)

        psi_vals = self.potential_network(X_expanded_for_Y, Y_expanded_for_X).squeeze(-1)
        cost_matrix = U_tensor @ Y_tensor.T

        slackness = cost_matrix - psi_vals
        max_slackness, _ = torch.max(slackness, dim=1, keepdim=True)
        slackness_stable = (slackness - max_slackness) / self.epsilon
        log_mean_exp = torch.logsumexp(slackness_stable, dim=1, keepdim=True) \
                - torch.log(torch.tensor(n, device=slackness.device, dtype=slackness.dtype))
        
        log_mean_exp += max_slackness / self.epsilon

        psi_estimate = self.epsilon * log_mean_exp

        return psi_estimate
    
    def gradient_inverse(self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor):
        requires_grad_backup, point_tensor.requires_grad = point_tensor.requires_grad, True
        inverse_tensor = torch.autograd.grad(
            self.potential_network(condition_tensor, point_tensor).sum(),
            point_tensor, create_graph=False
        )[0]
        point_tensor.requires_grad = requires_grad_backup
        return inverse_tensor.detach()

    def c_transform_inverse(self, condition_tensor: torch.Tensor, point_tensor: torch.Tensor):
        inverse_tensor = torch.nn.Parameter(torch.randn_like(point_tensor).contiguous())

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
            cost_matrix = torch.sum(point_tensor * inverse_tensor, dim=-1, keepdims=True)
            potential = self.potential_network(condition_tensor, inverse_tensor)
            slackness = (potential - cost_matrix).sum()
            slackness.backward()
            return slackness
        
        optimizer.step(slackness_closure)
        return inverse_tensor.detach()
    
    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        X_scaled = self.X_scaler(x)
        Y_scaled = self.Y_scaler(y)

        if self.potential_to_estimate_with_neural_network == "y":
            U_tensor = self.gradient_inverse(X_scaled, Y_scaled)
        else:
            if self.network_type in {"FFNN", "PICNN"}:
                error_message = f"Convergence is not guarenteed for {self.network_type} network"
                raise NotImplementedError(error_message)
            U_tensor = self.c_transform_inverse(X_scaled, Y_scaled)

        return U_tensor.requires_grad_(False).detach()

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        X_scaled = self.X_scaler(x)

        if self.potential_to_estimate_with_neural_network == "u":
            Y_tensor = self.gradient_inverse(X_scaled, u)
        else:
            if self.network_type in {"FFNN", "PICNN"}:
                error_message = f"Convergence is not guarenteed for {self.network_type} network"
                raise NotImplementedError(error_message)
            Y_tensor = self.c_transform_inverse(X_scaled, u)
        
        return (
            Y_tensor.requires_grad_(False)*torch.sqrt(self.Y_scaler.running_var) 
            + self.Y_scaler.running_mean
        ).detach()

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({
            "init_dict": self.init_dict,
            "state_dict": self.state_dict(),
            "class_name":"EntropicOTQuantileRegression"
        }, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self
    
    @classmethod
    def load_class(cls, path: str, map_location: torch.device = torch.device('cpu')) -> "EntropicOTQuantileRegression":
        data = torch.load(path, map_location=map_location)
        quadratic_potential = cls(**data["init_dict"])
        quadratic_potential.load_state_dict(data["state_dict"])
        return quadratic_potential