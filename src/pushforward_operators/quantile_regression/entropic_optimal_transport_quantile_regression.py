import torch.nn as nn
import torch
from tqdm import trange
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator

class EntropicOTQuantileRegression(PushForwardOperator, nn.Module):
    def __init__(self,
            feature_dimension: int,
            response_dimension: int,
            hidden_dimension: int,
            number_of_hidden_layers: int,
            epsilon: float,
            activation_function_name: str = "Softplus",
        ):
        super().__init__()
        self.activation_function_name = activation_function_name
        self.activation_function = getattr(nn, activation_function_name)()
        self.Y_scaler = nn.BatchNorm1d(response_dimension, affine=False)

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.psi_potential_network = nn.Sequential(
            nn.Linear(feature_dimension + response_dimension, hidden_dimension),
            self.activation_function,
            *hidden_layers,
            nn.Linear(hidden_dimension, 1)
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
        phi_potential_network_optimizer = torch.optim.AdamW(
            params=self.psi_potential_network.parameters(),
            **train_parameters.optimizer_parameters
        )
        if train_parameters.scheduler_parameters:
            phi_potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=phi_potential_network_optimizer, 
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            phi_potential_network_scheduler = None

        self.warmup_Y_scaler(dataloader)
        training_information = []
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    self.psi_potential_network.zero_grad()

                    Y_scaled_batch = self.Y_scaler(Y_batch)
                    U_batch = torch.randn_like(Y_scaled_batch)

                    psi = self.psi_potential_network(torch.cat([X_batch, Y_scaled_batch], dim=1))
                    phi = self.estimate_entropy_dual_phi(
                            X_tensor=X_batch,
                            U_tensor=U_batch,
                            Y_tensor=Y_scaled_batch
                    )
                    objective = torch.mean(phi) + torch.mean(psi)
                    objective.backward()

                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.psi_potential_network.parameters(), max_norm=10
                    ).item()

                    phi_potential_network_optimizer.step()
                    if phi_potential_network_scheduler is not None:
                        phi_potential_network_scheduler.step()

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
                                f", LR: {phi_potential_network_scheduler.get_last_lr()[0]:.6f}"
                                if phi_potential_network_scheduler is not None
                                else ""
                            )
                        )

        progress_bar.close()
        return self

    def estimate_entropy_dual_phi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates the entropy dual psi.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        n, _ = X_tensor.shape

        Y_expanded = Y_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_Y = X_tensor.unsqueeze(1).expand(-1, n, -1)
        XY = torch.cat((X_expanded_for_Y, Y_expanded), dim=-1)

        psi_vals = self.psi_potential_network(XY).squeeze(-1)
        cost_matrix = U_tensor @ Y_tensor.T

        slackness = cost_matrix - psi_vals
        max_slackness, _ = torch.max(slackness, dim=1, keepdim=True)
        slackness_stable = (slackness - max_slackness) / self.epsilon
        log_mean_exp = torch.logsumexp(slackness_stable, dim=1, keepdim=True) \
                - torch.log(torch.tensor(n, device=slackness.device, dtype=slackness.dtype))
        
        log_mean_exp += max_slackness / self.epsilon

        psi_estimate = self.epsilon * log_mean_exp

        return psi_estimate
    
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        self.Y_scaler.eval()
        requires_grad_backup = y.requires_grad
        y.requires_grad = True
        Y_scaled = self.Y_scaler(y)
        U = torch.autograd.grad(self.psi_potential_network(torch.cat([x, Y_scaled], dim=-1)).sum(), Y_scaled, create_graph=False)[0]
        y.requires_grad = requires_grad_backup
        return U.detach()

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({"state_dict": self.state_dict(), "epsilon": self.epsilon, "activation_function_name": self.activation_function_name}, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        self.epsilon = data["epsilon"]
        self.activation_function_name = data["activation_function_name"]
        return self