# Continuous Vector Quantile Regression Sanketh Vedulaon et al https://openreview.net/pdf?id=DUZbGAXcyL

import time
import torch
from tqdm.notebook import trange
from pushforward_operators import PushForwardOperator
from utils.distribution import sample_distribution_like
from infrastructure.classes import TrainParameters


class ICNN(torch.nn.Module):

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        output_dimension: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()

        Wzs = []
        Wzs.append(torch.nn.Linear(input_dimension, hidden_dimension))
        for _ in range(number_of_hidden_layers - 1):
            Wzs.append(torch.nn.Linear(hidden_dimension, hidden_dimension, bias=False))
        Wzs.append(torch.nn.Linear(hidden_dimension, 1, bias=False))
        self.Wzs = torch.nn.ModuleList(Wzs)

        Wxs = []
        for _ in range(number_of_hidden_layers - 1):
            Wxs.append(torch.nn.Linear(input_dimension, hidden_dimension))
        Wxs.append(torch.nn.Linear(input_dimension, output_dimension, bias=False))
        self.Wxs = torch.nn.ModuleList(Wxs)
        self.act = torch.nn.Softplus()

    def forward(self, x):
        z = self.act(self.Wzs[0](x))
        for Wz, Wx in zip(self.Wzs[1:-1], self.Wxs[:-1]):
            z = self.act(Wz(z) + Wx(x))
        return self.Wzs[-1](z) + self.Wxs[-1](x)


class FFNN(torch.nn.Module):

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        output_dimension: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.activation_function = torch.nn.ReLU()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(torch.nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.forward_network = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            torch.nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, input_tensor: torch.Tensor):
        return self.forward_network(input_tensor)


class VedulaQuantileNetwork(PushForwardOperator, torch.nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
    ):
        super().__init__()

        self.init_dict = {
            "feature_dimension": feature_dimension,
            "response_dimension": response_dimension,
            "hidden_dimension": hidden_dimension,
            "number_of_hidden_layers": number_of_hidden_layers,
        }
        self.model_information_dict = {
            "class_name": "VedulaQuantileNetwork",
        }

        self.linear_feature_network = torch.nn.Sequential(
            FFNN(
                input_dimension=feature_dimension,
                hidden_dimension=hidden_dimension,
                number_of_hidden_layers=number_of_hidden_layers,
                output_dimension=response_dimension,
            ),
            torch.nn.BatchNorm1d(
                num_features=response_dimension, affine=False, track_running_stats=True
            )
        )

        self.linear_potential_network = ICNN(
            input_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            output_dimension=response_dimension,
        )

        self.potential_network = ICNN(
            input_dimension=response_dimension,
            hidden_dimension=hidden_dimension,
            number_of_hidden_layers=number_of_hidden_layers,
            output_dimension=1,
        )

    def make_progress_bar_message(
        self, training_information: list[dict], epoch_idx: int
    ):
        running_mean_objective = sum(
            [
                information["potential_loss"]
                for information in training_information[-10:]
            ]
        ) / len(training_information[-10:])

        return (f"Epoch: {epoch_idx}, "
                f"Objective: {running_mean_objective:.3f}")

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

        potential_network_optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.potential_network.parameters()
                }, {
                    "params": self.linear_feature_network.parameters()
                }, {
                    "params": self.linear_potential_network.parameters()
                }
            ], **train_parameters.optimizer_parameters
        )

        training_information = []
        training_information_per_epoch = []

        progress_bar = trange(
            1, number_of_epochs_to_train + 1, desc="Training", disable=not verbose
        )

        for epoch_idx in progress_bar:
            start_of_epoch = time.perf_counter()
            potential_losses_per_epoch = []

            for batch_index, (X_batch, Y_batch) in enumerate(dataloader):
                U_batch = sample_distribution_like(Y_batch, "normal")

                potential_network_optimizer.zero_grad()
                phi = self.potential_network(U_batch)
                linear_features = self.linear_feature_network(X_batch)
                linear_potential = self.linear_potential_network(U_batch)

                discrete_c_transform_matrix = (
                    Y_batch @ U_batch.T - phi.T - linear_features @ linear_potential.T
                )

                psi = torch.logsumexp(discrete_c_transform_matrix, dim=1, keepdim=True)

                potential_network_objective = torch.mean(phi) + torch.mean(psi)
                potential_network_objective.backward()
                potential_network_optimizer.step()

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
                    progress_bar_message = self.make_progress_bar_message(
                        training_information=training_information, epoch_idx=epoch_idx
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
        self.model_information_dict["training_information"
                                    ] = training_information_per_epoch
        return self

    @torch.enable_grad()
    def push_y_given_x(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented("This method is not implemented.")

    @torch.enable_grad()
    def push_u_given_x(self, u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        point_tensor = u.clone()
        point_tensor.requires_grad = True

        linear_part_of_the_potential = (
            self.linear_feature_network(x) * self.linear_potential_network(u)
        ).sum(dim=1, keepdim=True)

        potential = (
            self.potential_network(point_tensor) + linear_part_of_the_potential
        )
        inverse_tensor = torch.autograd.grad(
            potential.sum(), point_tensor, create_graph=False
        )[0]
        return inverse_tensor.detach()

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
    ) -> "VedulaQuantileNetwork":
        data = torch.load(path, map_location=map_location)
        neural_quantile_regression = cls(**data["init_dict"])
        neural_quantile_regression.load_state_dict(data["state_dict"])
        neural_quantile_regression.model_information_dict = data.get(
            "model_information_dict", {}
        )
        return neural_quantile_regression
