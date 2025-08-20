from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch
import torch.nn as nn
from typing import Literal
from pushforward_operators.picnn import network_type_name_to_network_type

from tqdm import trange

class AmortizationNetwork(nn.Module):
    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str
    ):
        super().__init__()
        self.activation_function_name = activation_function_name
        self.activation_function = getattr(nn, activation_function_name)()
        self.feature_expansion_layer = nn.Linear(feature_dimension, response_dimension*2)
    
        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.amortization_network = nn.Sequential(
            nn.Linear(3*response_dimension, hidden_dimension),
            self.activation_function,
            *hidden_layers,
            nn.Linear(hidden_dimension, response_dimension)
        )

        self.identity_projection = nn.Linear(response_dimension, response_dimension)

    def forward(self, X: torch.Tensor, U: torch.Tensor):
        input_tensor = torch.cat([self.feature_expansion_layer(X), U], dim=-1)
        output_tensor = self.amortization_network(input_tensor)
        input_projection = self.identity_projection(U)
        return output_tensor + input_projection

class UnconstrainedAmortizedOTQuantileRegression(PushForwardOperator, nn.Module):
    def __init__(
        self,
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
        )

        self.amortization_network = AmortizationNetwork(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name,
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
        potential_network_optimizer = torch.optim.AdamW(self.potential_network.parameters(), **train_parameters.optimizer_parameters)
        amortization_network_optimizer = torch.optim.AdamW(self.amortization_network.parameters(), **train_parameters.optimizer_parameters)

        if train_parameters.scheduler_parameters:
            amortization_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(amortization_network_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
            potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(potential_network_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
        else:
            amortization_network_scheduler = None
            potential_network_scheduler = None

        training_information = []
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    U_batch = torch.randn_like(Y_batch)

                    if self.potential_to_estimate_with_neural_network == "y":
                        Y_amortized, U_amortized = self.amortization_network(X_batch, U_batch), None
                        Y_amortized_detached = Y_amortized.detach()
                        with torch.no_grad():
                            Y_batch_for_phi, U_batch_for_psi = self.push_u_given_x(
                                x=X_batch,
                                u=U_batch,
                                y_initial=Y_amortized_detached
                            ), None

                        self.amortization_network.zero_grad()  
                        amortization_network_objective = torch.norm(Y_amortized - Y_batch_for_phi, dim=-1).mean()
                        amortization_network_objective.backward()
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            self.amortization_network.parameters(), max_norm=10
                        ).item()
                        amortization_network_optimizer.step()
                            
                    else:
                        U_amortized = self.amortization_network(X_batch, Y_batch)
                        U_amortized_detached = U_amortized.detach()
                        with torch.no_grad():
                            U_batch_for_psi, Y_batch_for_phi = self.push_y_given_x(
                                x=X_batch,
                                y=Y_batch,
                                u_initial=U_amortized_detached
                            ), None

                        self.amortization_network.zero_grad()  
                        amortization_network_objective = torch.norm(U_amortized - U_batch_for_psi, dim=-1).mean()
                        amortization_network_objective.backward()
                        torch.nn.utils.clip_grad.clip_grad_norm_(
                            self.amortization_network.parameters(), max_norm=10
                        ).item()
                        amortization_network_optimizer.step()

                    potential_network_optimizer.zero_grad()
                    psi = self.estimate_psi(X_tensor=X_batch, Y_tensor=Y_batch, U_tensor=U_batch_for_psi)
                    phi = self.estimate_phi(X_tensor=X_batch, U_tensor=U_batch, Y_tensor=Y_batch_for_phi)
                    potential_network_objective = torch.mean(phi) + torch.mean(psi)
                    potential_network_objective.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.potential_network.parameters(), max_norm=10
                    ).item()
                    potential_network_optimizer.step()

                    if potential_network_scheduler is not None and amortization_network_scheduler is not None:
                        potential_network_scheduler.step()
                        amortization_network_scheduler.step()

                    if verbose:
                        training_information.append({
                                "potential": potential_network_objective.item(),
                                "amortization": amortization_network_objective.item(),
                                "epoch_index": epoch_idx
                        })

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
                            f", LR: {potential_network_scheduler.get_last_lr()[0]:.6f}"
                            if potential_network_scheduler is not None
                            else ""
                        )
                        
                        progress_bar.set_description(description_message)

        progress_bar.close()
        return self
    
    def estimate_psi(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, U_tensor: torch.Tensor | None = None):
        """Estimates psi, either with Neural Network or by solving optimization with sgd.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "y":
            return self.potential_network(X_tensor, Y_tensor)
        else:
            return torch.sum(Y_tensor*U_tensor, dim=-1, keepdim=True) - self.potential_network(X_tensor, U_tensor)
    
    def estimate_phi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_tensor: torch.Tensor | None = None):
        """Estimates phi, either with Neural Network or entropic dual.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        if self.potential_to_estimate_with_neural_network == "u":
            return self.potential_network(X_tensor, U_tensor)
        else:
            return torch.sum(Y_tensor*U_tensor, dim=-1, keepdim=True) - self.potential_network(X_tensor, Y_tensor)

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor, u_initial: torch.Tensor | None = None) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        if self.potential_to_estimate_with_neural_network == "y":
            requires_grad_backup, y.requires_grad = y.requires_grad, True
            U_tensor = torch.autograd.grad(self.potential_network(x, y).sum(), y, create_graph=False)[0]
            y.requires_grad = requires_grad_backup
            return U_tensor.detach()
    
        if u_initial is None:
            u_initial = self.amortization_network(x, y)

        U_tensor = torch.nn.Parameter(u_initial.clone().contiguous())

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
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor, y_initial: torch.Tensor | None = None) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        if self.potential_to_estimate_with_neural_network == "u":
            requires_grad_backup, u.requires_grad = u.requires_grad, True
            Y_tensor = torch.autograd.grad(self.potential_network(x, u).sum(), u, create_graph=False)[0]
            u.requires_grad = requires_grad_backup
            return Y_tensor.detach()
    
        if y_initial is None:
            y_initial = self.amortization_network(x, u)

        Y_tensor = torch.nn.Parameter(y_initial.clone().contiguous())

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
            "class_name":"UnconstrainedAmortizedOTQuantileRegression"
        }, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self
    
    @classmethod
    def load_class(cls, path: str, map_location: torch.device = torch.device('cpu')) -> "UnconstrainedAmortizedOTQuantileRegression":
        data = torch.load(path, map_location=map_location)
        quadratic_potential = cls(**data["init_dict"])
        quadratic_potential.load_state_dict(data["state_dict"])
        return quadratic_potential