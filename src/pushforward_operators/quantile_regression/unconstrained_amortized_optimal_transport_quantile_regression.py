from pushforward_operators.protocol import PushForwardOperator
from infrastructure.classes import TrainParameters
import torch
import torch.nn as nn
from tqdm import trange

class PotentialNN(nn.Module):
    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str,
    ):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(0.))
        self.activation_function_name = activation_function_name
        self.activation_function = getattr(nn, activation_function_name)()
        self.feature_expansion_layer = nn.Linear(feature_dimension, response_dimension*2)

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.potential_network = nn.Sequential(
            nn.Linear(3*response_dimension, hidden_dimension),
            self.activation_function,
            *hidden_layers,
            nn.Linear(hidden_dimension, 1)
        )

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
        input_tensor = torch.cat([self.feature_expansion_layer(X), Y], dim=-1)
        output_tensor = self.potential_network(input_tensor)
        return output_tensor + torch.exp(self.log_alpha) * ( 0.5 * torch.norm(Y, dim=-1, keepdim=True)**2 )

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
    def __init__(self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str,
    ):
        super().__init__()
        self.init_dict = {
            "class_name": "UnconstrainedAmortizedOTQuantileRegression",
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "activation_function_name": activation_function_name
        }

        self.psi_potential_network = PotentialNN(
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
        psi_potential_network_optimizer = torch.optim.AdamW(self.psi_potential_network.parameters(), **train_parameters.optimizer_parameters)
        amortization_network_optimizer = torch.optim.AdamW(self.amortization_network.parameters(), **train_parameters.optimizer_parameters)

        if train_parameters.scheduler_parameters:
            amortization_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(amortization_network_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
            psi_potential_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(psi_potential_network_optimizer, total_number_of_optimizer_steps, **train_parameters.scheduler_parameters)
        else:
            psi_potential_network_scheduler = None
            amortization_network_scheduler = None

        training_information = []
        progress_bar = trange(1, number_of_epochs_to_train+1, desc="Training", disable=not verbose)

        for epoch_idx in progress_bar:
                for X_batch, Y_batch in dataloader:
                    U_batch = torch.randn_like(Y_batch)

                    Y_amortized = self.amortization_network(X_batch, U_batch)
                    Y_amortized_detached = Y_amortized.detach()

                    with torch.no_grad():
                        Y_batch_for_phi = self.estimate_Y_from_psi(
                            X_tensor=X_batch,
                            U_tensor=U_batch,
                            Y_init=Y_amortized_detached
                        )

                    self.psi_potential_network.zero_grad()
                    psi = self.psi_potential_network(X_batch, Y_batch)
                    phi = torch.sum(Y_batch_for_phi * U_batch, dim=-1, keepdims=True) \
                            - self.psi_potential_network(X_batch, Y_batch_for_phi)
                    potential_network_objective = torch.mean(phi) + torch.mean(psi)
                    potential_network_objective.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.psi_potential_network.parameters(), max_norm=10
                    ).item()
                    psi_potential_network_optimizer.step()


                    self.amortization_network.zero_grad()  
                    amortization_network_objective = torch.norm(Y_amortized - Y_batch_for_phi, dim=-1).mean()
                    amortization_network_objective.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(
                        self.amortization_network.parameters(), max_norm=10
                    ).item()
                    amortization_network_optimizer.step()

                    if psi_potential_network_scheduler is not None and amortization_network_scheduler is not None:
                        psi_potential_network_scheduler.step()
                        amortization_network_scheduler.step()

                    if verbose:
                        training_information.append({
                                "potential": potential_network_objective.item(),
                                "amortization": amortization_network_objective.item(),
                                "epoch_index": epoch_idx
                        })

                        running_mean_potential_objective = sum([information["potential"] for information in training_information[-10:]]) / len(training_information[-10:])
                        running_mean_amortization_objective = sum([information["amortization"] for information in training_information[-10:]]) / len(training_information[-10:])
                        progress_bar.set_description(
                            (
                                f"Epoch: {epoch_idx}, "
                                f"Potential Objective: {running_mean_potential_objective:.3f}, "
                                f"Amortization Objective: {running_mean_amortization_objective:.3f}"
                            ) + \
                            (
                                f", LR: {psi_potential_network_scheduler.get_last_lr()[0]:.6f}"
                                if psi_potential_network_scheduler is not None
                                else ""
                            )
                        )

        progress_bar.close()
        return self

    def estimate_Y_from_psi(self, X_tensor: torch.Tensor, U_tensor: torch.Tensor, Y_init: torch.Tensor | None = None, verbose: bool = False):
            """
            Estimate U tensor by minimizing u^T y - phi(x, u) for given x and y.
            phi(x, u) is assume to be a potential function convex in u.

            Args:
            X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
            Y_tensor (torch.Tensor): The tensor of oversampled variables y, with shape [n, q].

            Returns:
            torch.Tensor: A scalar tensor representing the estimated phi value.
            """
            if Y_init is not None:
                 Y_tensor = Y_init.detach().clone().requires_grad_(True)
            else:
                Y_tensor = torch.randn_like(U_tensor).requires_grad_(True)

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
                cost_matrix = torch.sum(U_tensor * Y_tensor, dim=-1, keepdims=True)
                psi_potential = self.psi_potential_network(X_tensor, Y_tensor)
                slackness = (psi_potential - cost_matrix).sum()
                slackness.backward()
                return slackness

            optimizer.step(slackness_closure)

            if verbose:
                optimal_Y_tensor_potential = self.psi_potential_network(X_tensor, Y_tensor).sum()
                approximated_U_tensor = torch.autograd.grad(optimal_Y_tensor_potential.sum(), Y_tensor)[0]
                estimation_error = (approximated_U_tensor - U_tensor)
                print(f"Maximal dual problem vector approximation error: {estimation_error.abs().max().item()}")

            return Y_tensor.detach()

    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Generates Y|X by applying a push forward operator to U.
        """
        requires_grad_backup = y.requires_grad
        y.requires_grad = True
        pushforward_of_u = torch.autograd.grad(self.psi_potential_network(x, y).sum(), y, create_graph=False)[0]
        y.requires_grad = requires_grad_backup
        return pushforward_of_u

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save({"init_dict": self.init_dict, "state_dict": self.state_dict()}, path)

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        """Loads the pushforward operator from a file.

        Args:
            path (str): Path to load the pushforward operator from.
        """
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        self.init_dict = data["init_dict"]
        return self