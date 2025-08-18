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

    def warmup_Y_scaler(self, dataloader: torch.utils.data.DataLoader, num_passes: int = 1):
        """Run over the data (no grad) to populate BatchNorm running stats."""
        self.Y_scaler.train()
        with torch.no_grad():
            for _ in range(num_passes):
                for _, Y in dataloader:
                    _ = self.Y_scaler(Y)
        self.Y_scaler.eval()

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

        self.warmup_Y_scaler(dataloader)
        training_information = []
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    self.potential_network.zero_grad()

                    Y_scaled_batch = self.Y_scaler(Y_batch)
                    U_batch = torch.randn_like(Y_scaled_batch)

                    psi = self.estimate_psi(
                            X_tensor=X_batch,
                            U_tensor=U_batch,
                            Y_tensor=Y_scaled_batch
                    )
                    phi = self.estimate_phi(
                            X_tensor=X_batch,
                            U_tensor=U_batch,
                            Y_tensor=Y_scaled_batch
                    )
                    
                    objective = torch.mean(phi) + torch.mean(psi)
                    objective.backward()

                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.potential_network.parameters(), max_norm=10
                    ).item()

                    potential_network_optimizer.step()
                    if potential_network_scheduler is not None:
                        potential_network_scheduler.step()

                    if verbose:
                        training_information.append({
                                "objective": objective.item(),
                                "epoch_index": epoch_idx
                        })

                        running_mean_objective = sum([information["objective"] for information in training_information[-10:]]) / len(training_information[-10:])
                        progress_bar.set_description(
                            (
                                f"Epoch: {epoch_idx}, "
                                f"Objective: {running_mean_objective:.3f}"
                            ) + \
                            (
                                f", LR: {potential_network_scheduler.get_last_lr()[0]:.6f}"
                                if potential_network_scheduler is not None
                                else ""
                            )
                        )

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
    
    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        self.Y_scaler.eval()
        if self.potential_to_estimate_with_neural_network == "y":
            requires_grad_backup, y.requires_grad = y.requires_grad, True
            Y_scaled = self.Y_scaler(y)
            potential_value = self.potential_network(x, Y_scaled).sum()

            U_tensor = torch.autograd.grad(potential_value, Y_scaled, create_graph=False)[0]
            y.requires_grad = requires_grad_backup
            return U_tensor.detach()
        else:
            if self.network_type in {"SCFFNN", "FFNN"}:
                error_message = f"Convergence is not guarenteed for {self.network_type} network"
                raise NotImplementedError(error_message)
            
            U_init = torch.randn_like(y)
            U_tensor = torch.nn.Parameter(U_init.clone().contiguous())

            X_tensor = x.clone()
            Y_tensor = self.Y_scaler(y)

            optimizer = torch.optim.LBFGS(
                [U_tensor],
                lr=0.01,
                line_search_fn="strong_wolfe",
                max_iter=1000,
                tolerance_grad=1e-10,
                tolerance_change=1e-10
            )

            def slackness_closure():
                optimizer.zero_grad()
                potential = self.potential_network(X_tensor, U_tensor)
                objective = (potential - torch.sum(Y_tensor*U_tensor, dim=-1, keepdim=True)).sum()
                objective.backward()
                return objective

            optimizer.step(slackness_closure)

            return U_tensor.requires_grad_(False).detach()

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        self.Y_scaler.eval()

        if self.potential_to_estimate_with_neural_network == "u":
            requires_grad_backup, u.requires_grad = u.requires_grad, True
            potential_value = self.potential_network(x, u).sum()

            Y_scaled = torch.autograd.grad(potential_value, u, create_graph=False)[0]
            Y_tensor = Y_scaled*torch.sqrt(self.Y_scaler.running_var) + self.Y_scaler.running_mean
            u.requires_grad = requires_grad_backup

            return Y_tensor.detach()
    
        else:
            if self.network_type in {"SCFFNN", "FFNN"}:
                error_message = f"Convergence is not guarenteed for {self.network_type} network"
                raise NotImplementedError(error_message)
            
            Y_init = torch.randn_like(u)
            Y_tensor = torch.nn.Parameter(Y_init.clone().contiguous())
            U_tensor = u.clone()
            X_tensor = x.clone()

            optimizer = torch.optim.LBFGS(
                [Y_tensor],
                lr=1,
                line_search_fn="strong_wolfe",
                max_iter=1000,
                tolerance_grad=1e-10,
                tolerance_change=1e-10
            )

            def slackness_closure():
                optimizer.zero_grad()
                potential = self.potential_network(X_tensor, Y_tensor)
                objective = (potential - torch.sum(Y_tensor*U_tensor, dim=-1, keepdim=True)).sum()
                objective.backward()
                return objective
            
            optimizer.step(slackness_closure)

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
    def load(cls, path: str, map_location: torch.device = torch.device('cpu')) -> "EntropicOTQuantileRegression":
        data = torch.load(path, map_location=map_location)
        quadratic_potential = cls(**data["init_dict"])
        quadratic_potential.load_state_dict(data["state_dict"])
        return quadratic_potential