import torch.nn as nn
import torch
from geomloss import SamplesLoss
from tqdm import trange
from typing import Literal
from infrastructure.classes import TrainParameters
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators.picnn import FFNN


class MongeMapNetwork(nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str = "Softplus",
    ):
        super().__init__()
        self.monge_map_network = FFNN(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name,
            output_dimension=response_dimension
        )

    def pushforward(
        self, condition: torch.Tensor, tensor: torch.Tensor
    ) -> torch.Tensor:
        return tensor - self.monge_map_network(condition, tensor)

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
        return self.monge_map_network(condition, tensor)


class MongeGapTransport(PushForwardOperator, nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        activation_function_name: str = "Softplus",
        potential_to_estimate_with_neural_network: Literal["y", "u"] = "y",
        jacobian_weight: float = 1e-1,
        c_optimality_weight: float = 1e-1
    ):
        super().__init__()
        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
            "activation_function_name": activation_function_name,
            "potential_to_estimate_with_neural_network":
            potential_to_estimate_with_neural_network,
            "jacobian_weight": jacobian_weight,
            "c_optimality_weight": c_optimality_weight
        }
        self.potential_to_estimate_with_neural_network = potential_to_estimate_with_neural_network
        self.monge_map_network = MongeMapNetwork(
            feature_dimension=feature_dimension,
            response_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            activation_function_name=activation_function_name
        )
        self.sinkhorn_divergence = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
        self.jacobian_weight = jacobian_weight
        self.c_optimality_weight = c_optimality_weight

    def make_progress_bar_message(
        self, training_information: list[dict], epoch_idx: int,
        last_learning_rate: float | None
    ):
        running_mean_objective = sum(
            [information["objective"] for information in training_information[-10:]]
        ) / len(training_information[-10:])

        return  (
            f"Epoch: {epoch_idx}, "
            f"Objective: {running_mean_objective:.3f}"
        ) + \
        (
            f", LR: {last_learning_rate[0]:.6f}"
            if last_learning_rate is not None
            else ""
        )

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
        monge_map_network_optimizer = torch.optim.AdamW(
            params=self.monge_map_network.parameters(),
            **train_parameters.optimizer_parameters
        )
        if train_parameters.scheduler_parameters:
            monge_map_network_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=monge_map_network_optimizer,
                T_max=total_number_of_optimizer_steps,
                **train_parameters.scheduler_parameters
            )
        else:
            monge_map_network_scheduler = None

        training_information = []
        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )
        X_previous, Y_previous = next(iter(dataloader))

        for epoch_idx in progress_bar:
            progress_bar.set_description(f"Epoch {epoch_idx}")
            for X_batch, Y_batch in dataloader:

                U_batch = torch.randn_like(Y_batch)

                if self.potential_to_estimate_with_neural_network == "y":
                    Y_pushforward = self.monge_map_network.pushforward(X_batch, Y_batch)
                    fitting_cost = self.sinkhorn_divergence(U_batch, Y_pushforward)
                else:
                    U_pushforward = self.monge_map_network.pushforward(X_batch, U_batch)
                    fitting_cost = self.sinkhorn_divergence(U_pushforward, Y_batch)

                if self.potential_to_estimate_with_neural_network == "y":
                    condition_sample_for_c_optimality = X_previous[:X_batch.shape[0]]
                    tensor_sample_for_c_optimality = Y_previous[:X_batch.shape[0]]
                    if X_batch.shape[0] == X_previous.shape[0]:
                        X_previous, Y_previous = X_batch.clone(), Y_batch.clone()
                else:
                    condition_sample_for_c_optimality = X_previous[:U_batch.shape[0]]
                    tensor_sample_for_c_optimality = torch.randn_like(U_batch)
                    if X_batch.shape[0] == X_previous.shape[0]:
                        X_previous = X_batch.clone()

                pairwise_distances = torch.cdist(
                    condition_sample_for_c_optimality, condition_sample_for_c_optimality
                )
                _, neighbor_indices = torch.topk(
                    pairwise_distances, 10, dim=1, largest=False
                )

                condition_sample_for_c_optimality_neighbor_groups = \
                    condition_sample_for_c_optimality[neighbor_indices]

                tensor_sample_for_c_optimality_neighbor_groups = \
                    tensor_sample_for_c_optimality[neighbor_indices]

                pushed_tensor_for_c_optimality_neighbor_groups = self.monge_map_network.pushforward(
                    condition_sample_for_c_optimality_neighbor_groups,
                    tensor_sample_for_c_optimality_neighbor_groups
                )

                c_optimality_cost_term = (
                    tensor_sample_for_c_optimality_neighbor_groups -
                    pushed_tensor_for_c_optimality_neighbor_groups
                ).norm(dim=-1).pow(2).mean(dim=1)

                c_optimality_sinkhorn_term = self.sinkhorn_divergence(
                    tensor_sample_for_c_optimality_neighbor_groups,
                    pushed_tensor_for_c_optimality_neighbor_groups
                )

                c_optimality_cost = (
                    c_optimality_cost_term - c_optimality_sinkhorn_term
                ).mean()

                if self.potential_to_estimate_with_neural_network == "y":
                    input_for_jacobian = Y_batch
                else:
                    input_for_jacobian = U_batch

                f_single = lambda x, y: self.monge_map_network(
                    x.unsqueeze(0), y.unsqueeze(0)
                ).squeeze(0)
                jacobian_function = torch.func.jacrev(f_single, argnums=1)
                vectorized_jacobian_function = torch.vmap(jacobian_function)
                jacobian: torch.Tensor = vectorized_jacobian_function(
                    X_batch, input_for_jacobian
                )
                jacobian_cost = torch.mean(
                    torch.norm(jacobian - jacobian.transpose(1, 2), dim=(1, 2))
                )

                monge_map_network_optimizer.zero_grad()
                monge_map_network_objective: torch.Tensor = (
                    fitting_cost + self.jacobian_weight * jacobian_cost +
                    self.c_optimality_weight * c_optimality_cost
                )
                monge_map_network_objective.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    self.monge_map_network.parameters(), max_norm=10
                )
                monge_map_network_optimizer.step()

                if monge_map_network_scheduler is not None:
                    monge_map_network_scheduler.step()

                if verbose:
                    training_information.append(
                        {"objective": monge_map_network_objective.item()}
                    )

                    running_mean_objective = sum(
                        [info["objective"] for info in training_information[-10:]]
                    ) / len(training_information[-10:])

                    postfix_dict = {"Objective": f"{running_mean_objective:.3f}"}
                    if monge_map_network_scheduler is not None:
                        last_lr = monge_map_network_scheduler.get_last_lr()[0]
                        postfix_dict["LR"] = f"{last_lr:.6f}"

                    progress_bar.set_postfix(postfix_dict)

        progress_bar.close()
        return self

    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes y variable to the latent space given condition x"""
        if self.potential_to_estimate_with_neural_network == "y":
            return self.monge_map_network.pushforward(x, y)
        else:
            raise NotImplementedError("Not implemented")

    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Pushes u variable to the y space given condition x"""
        if self.potential_to_estimate_with_neural_network == "u":
            return self.monge_map_network.pushforward(x, u)
        else:
            raise NotImplementedError("Not implemented")

    def save(self, path: str):
        """Saves the pushforward operator to a file.

        Args:
            path (str): Path to save the pushforward operator.
        """
        torch.save(
            {
                "init_dict": self.init_dict,
                "state_dict": self.state_dict(),
                "class_name": "MongeGapTransport"
            }, path
        )

    def load(self, path: str, map_location: torch.device = torch.device('cpu')):
        data = torch.load(path, map_location=map_location)
        self.load_state_dict(data["state_dict"])
        return self

    @classmethod
    def load_class(
        cls, path: str, map_location: torch.device = torch.device('cpu')
    ) -> "MongeGapTransport":
        data = torch.load(path, map_location=map_location)
        monge_gap_transport = cls(**data["init_dict"])
        monge_gap_transport.load_state_dict(data["state_dict"])
        return monge_gap_transport
