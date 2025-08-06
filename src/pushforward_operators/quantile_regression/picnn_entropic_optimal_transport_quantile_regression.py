import torch.nn as nn
import torch
from tqdm import trange
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators.picnn import PICNN

class PICNNEntropicOTQuantileRegression(PushForwardOperator, nn.Module):
    def __init__(self,
        x_dimension: int,
        y_dimension: int,
        u_dimension: int,
        z_dimension: int,
        number_of_hidden_layers: int = 1,
        epsilon: float = 1e-7,
        number_of_samples_for_entropy_dual_estimation: int = 2048
    ):
        super().__init__()
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.u_dimension = u_dimension
        self.z_dimension = z_dimension
        self.number_of_hidden_layers = number_of_hidden_layers
        self.epsilon = epsilon
        self.number_of_samples_for_entropy_dual_estimation = number_of_samples_for_entropy_dual_estimation
        self.Y_scaler = nn.BatchNorm1d(y_dimension, affine=False)

        self.phi_potential_network = PICNN(
            x_dimension=x_dimension,
            y_dimension=y_dimension,
            u_dimension=u_dimension,
            z_dimension=z_dimension,
            output_dimension=1,
            number_of_hidden_layers=number_of_hidden_layers
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
        phi_potential_network_optimizer = torch.optim.AdamW(self.phi_potential_network.parameters(), **train_parameters.optimizer_parameters)
        if train_parameters.scheduler_parameters:
            phi_potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(phi_potential_network_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
        else:
            phi_potential_network_scheduler = None

        training_information = []
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    self.phi_potential_network.zero_grad()

                    Y_scaled_batch = self.Y_scaler(Y_batch)
                    U_batch = torch.randn_like(Y_scaled_batch)

                    phi = self.phi_potential_network(X_batch, U_batch)
                    psi = self.estimate_entropy_dual_psi(
                            X_tensor=X_batch,
                            U_tensor=torch.randn(
                                    self.number_of_samples_for_entropy_dual_estimation, Y_batch.shape[1],
                                    **{"device": Y_batch.device, "dtype": Y_batch.dtype}
                            ),
                            Y_tensor=Y_scaled_batch
                    )
                    objective = torch.mean(phi) + torch.mean(psi)

                    objective.backward()
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

    def estimate_entropy_dual_psi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_tensor: torch.Tensor):
        """Estimates the entropy dual psi.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            U_tensor (torch.Tensor): Random variable to be pushed forward.
            Y_tensor (torch.Tensor): Output tensor.
        """
        n, _ = X_tensor.shape
        m, _ = U_tensor.shape

        U_expanded = U_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_U = X_tensor.unsqueeze(1).expand(-1, m, -1)

        phi_vals = self.phi_potential_network(X_expanded_for_U, U_expanded).squeeze(-1)
        cost_matrix = Y_tensor @ U_tensor.T

        slackness = cost_matrix - phi_vals

        log_mean_exp = torch.logsumexp(slackness / self.epsilon, dim=1, keepdim=True) \
                - torch.log(torch.tensor(m, device=slackness.device, dtype=slackness.dtype))

        psi_estimate = self.epsilon * log_mean_exp

        return psi_estimate

    def push_forward_u_given_x(self, U: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """Generates Y|X by applying a push forward operator to U.
        """
        requires_grad_backup = U.requires_grad
        U.requires_grad = True
        pushforward_of_u = torch.autograd.grad(self.phi_potential_network(X, U).sum(), U, create_graph=False)[0]
        pushforward_of_u = pushforward_of_u * torch.sqrt(self.Y_scaler.running_var) + self.Y_scaler.running_mean
        U.requires_grad = requires_grad_backup
        return pushforward_of_u

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({"state_dict": self.state_dict(), "epsilon": self.epsilon}, path)

    def load(self, path: str, map_location: torch.device = torch.device("cpu")):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        self.epsilon = data["epsilon"]
        return self