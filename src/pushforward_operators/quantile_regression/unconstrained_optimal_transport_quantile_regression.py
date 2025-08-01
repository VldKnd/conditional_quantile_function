from pushforward_operators.protocol import PushForwardOperator
from infrastructure.dataclasses import TrainParameters
import torch
import torch.nn as nn
from tqdm import trange
from pushforward_operators.picnn import SCPICNN

class UnconstrainedOTQuantileRegression(PushForwardOperator, nn.Module):
    def __init__(self,
        alpha: float,
        x_dimension: int,
        y_dimension: int,
        u_dimension: int,
        z_dimension: int,
        number_of_hidden_layers: int
    ):
        super().__init__()
        self.init_dict = {
            "class_name": "UnconstrainedOTQuantileRegression",
            "alpha": alpha,
            "x_dimension": x_dimension,
            "y_dimension": y_dimension,
            "u_dimension": u_dimension,
            "z_dimension": z_dimension,
            "number_of_hidden_layers": number_of_hidden_layers
        }

        self.Y_scaler = nn.BatchNorm1d(y_dimension, affine=False)

        self.phi_potential_network = SCPICNN(
            alpha=alpha,
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
                    U_batch = torch.randn_like(Y_batch)
                    Y_batch_scaled = self.Y_scaler(Y_batch)

                    U_batch_for_psi = self.estimate_U_from_phi(
                            X_tensor=X_batch,
                            Y_tensor=Y_batch_scaled,
                            verbose=False,
                    )

                    self.phi_potential_network.zero_grad()

                    phi = self.phi_potential_network(X_batch, U_batch)
                    psi = torch.sum(U_batch_for_psi * Y_batch_scaled, dim=-1, keepdims=True) \
                            - self.phi_potential_network(X_batch, U_batch_for_psi)
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

    def estimate_U_from_phi(self, X_tensor: torch.Tensor, Y_tensor: torch.Tensor, verbose: bool = False):
            """
            Estimate U tensor by minimizing u^T y - phi(x, u) for given x and y.
            phi(x, u) is assume to be a potential function convex in u.

            Args:
            X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
            Y_tensor (torch.Tensor): The tensor of oversampled variables y, with shape [n, q].

            Returns:
            torch.Tensor: A scalar tensor representing the estimated phi value.
            """
            U_tensor = torch.randn_like(Y_tensor)
            U_tensor.requires_grad = True

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
                cost_matrix = torch.sum(U_tensor * Y_tensor, dim=-1, keepdims=True)
                phi_potential = self.phi_potential_network(X_tensor, U_tensor)
                slackness = (phi_potential - cost_matrix).sum()
                slackness.backward()
                return slackness

            optimizer.step(slackness_closure)

            if verbose:
                optimal_U_tensor_potential = self.phi_potential_network(X_tensor, U_tensor).sum()
                approximated_Y_tensor = torch.autograd.grad(optimal_U_tensor_potential.sum(), U_tensor)[0]
                estimation_error = (approximated_Y_tensor - Y_tensor)
                print(f"Maximal dual problem vector approximation error: {estimation_error.abs().max().item()}")

            return U_tensor

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
        torch.save({"init_dict": self.init_dict, "state_dict": self.state_dict()}, path)

    def load(self, path: str):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path)
        self.load_state_dict(data["state_dict"])
        self.init_dict = data["init_dict"]
        return self