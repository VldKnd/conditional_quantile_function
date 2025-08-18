from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch
import torch.nn as nn
from tqdm import trange
from typing import Literal
from pushforward_operators.picnn import network_type_name_to_network_type

class UnconstrainedOTQuantileRegression(PushForwardOperator, nn.Module):
    def __init__(self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str,
        network_type: Literal["SCFFNN", "PISCNN"] = "SCFFNN",
        potential_to_estimate_with_neural_network: Literal["y", "u"] = "y",
    ):
        super().__init__()

        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "activation_function_name": activation_function_name,
            "network_type":network_type,
            "potential_to_estimate_with_neural_network":potential_to_estimate_with_neural_network
        }

        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network
        self.potential_network = network_type_name_to_network_type[network_type](
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name,
            output_dimension=1
        )


    def fit(self, dataloader: torch.utils.data.DataLoader, train_parameters: TrainParameters, *args, **kwargs):
        """Fits the pushforward operator to the data.

        Args:
            dataloader (torch.utils.data.DataLoader): Data loader.
            train_parameters (TrainParameters): Training parameters.
        """
        number_of_epochs_to_train = train_parameters.number_of_epochs_to_train
        number_of_epochs_to_train = 1
        verbose = train_parameters.verbose
        total_number_of_optimizer_steps = number_of_epochs_to_train * len(dataloader)
        potential_network_optimizer = torch.optim.AdamW(self.potential_network.parameters(), **train_parameters.optimizer_parameters)
        if train_parameters.scheduler_parameters:
            potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(potential_network_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
        else:
            potential_network_scheduler = None

        training_information = []
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    U_batch = torch.randn_like(Y_batch)

                    psi = self.estimate_psi(
                        X_tensor=X_batch,
                        Y_tensor=Y_batch,
                    )
                    phi = self.estimate_phi(
                        X_tensor=X_batch,
                        U_tensor=U_batch,
                    )

                    potential_network_optimizer.zero_grad()
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

    def estimate_psi(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates psi, either with Neural Network or by solving optimization with sgd.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "y":
            return self.potential_network(X_tensor, Y_tensor)

        U_tensor = self.push_y_given_x(y=Y_tensor, x=X_tensor)
        return torch.sum(Y_tensor*U_tensor, dim=-1, keepdim=True) - self.potential_network(X_tensor, U_tensor)
    
    def estimate_phi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor):
        """Estimates phi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "u":
            return self.potential_network(X_tensor, U_tensor)

        Y_tensor = self.push_u_given_x(u=U_tensor, x=X_tensor)
        return torch.sum(Y_tensor*U_tensor, dim=-1, keepdim=True) - self.potential_network(X_tensor, Y_tensor)

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        if self.potential_to_estimate_with_neural_network == "y":
            requires_grad_backup, y.requires_grad = y.requires_grad, True
            potential_value = self.potential_network(x, y).sum()

            U_tensor = torch.autograd.grad(potential_value, y, create_graph=False)[0]
            y.requires_grad = requires_grad_backup
            return U_tensor.detach()
        
        U_init = torch.randn_like(y)
        U_tensor = torch.nn.Parameter(U_init.clone().contiguous())

        optimizer = torch.optim.LBFGS(
            [U_tensor],
            lr=1,
            line_search_fn="strong_wolfe",
            max_iter=1000,
            tolerance_grad=1e-10,
            tolerance_change=1e-10
        )

        def slackness_closure():
            optimizer.zero_grad()
            cost_matrix = torch.sum(y * U_tensor, dim=-1, keepdims=True)
            psi_potential = self.potential_network(x, U_tensor)
            slackness = (psi_potential - cost_matrix).sum()
            slackness.backward()
            return slackness

        optimizer.step(slackness_closure)

        return U_tensor.detach()

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        if self.potential_to_estimate_with_neural_network == "u":
            requires_grad_backup, u.requires_grad = u.requires_grad, True
            potential_value = self.potential_network(x, u).sum()
            Y_tensor = torch.autograd.grad(potential_value, u, create_graph=False)[0]
            u.requires_grad = requires_grad_backup
            return Y_tensor.detach()
        
        Y_init = torch.randn_like(u)
        Y_tensor = torch.nn.Parameter(Y_init.clone().contiguous())

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
            cost_matrix = torch.sum(u * Y_tensor, dim=-1, keepdims=True)
            psi_potential = self.potential_network(x, Y_tensor)
            slackness = (psi_potential - cost_matrix).sum()
            slackness.backward()
            return slackness

        optimizer.step(slackness_closure)

        return Y_tensor.detach()

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({
            "init_dict": self.init_dict,
            "state_dict": self.state_dict(),
            "class_name":"UnconstrainedOTQuantileRegression"
        }, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self
    
    @classmethod
    def load(cls, path: str, map_location: torch.device = torch.device('cpu')) -> "UnconstrainedOTQuantileRegression":
        data = torch.load(path, map_location=map_location)
        quadratic_potential = cls(**data["init_dict"])
        quadratic_potential.load_state_dict(data["state_dict"])
        return quadratic_potential