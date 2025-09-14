import torch
import math


class ActNorm(torch.nn.Module):
    """Adapted from https://github.com/ludvb/actnorm/blob/master/actnorm/actnorm.py"""

    def __init__(self, feature_dimension: int):
        super().__init__()
        self.epsilon = 1e-8
        self.scale = torch.nn.Parameter(torch.zeros(feature_dimension))
        self.bias = torch.nn.Parameter(torch.zeros(feature_dimension))
        self.register_buffer("initialized", torch.tensor(False))

    def initialize(self, x: torch.Tensor):
        reduce_dims = tuple(range(x.dim() - 1))
        x_detached = x.detach()
        data_std = x_detached.std(dim=reduce_dims, unbiased=False)
        data_scale = 1 / (data_std + self.epsilon)
        data_scaled_mean = x_detached.mean(dim=reduce_dims) * data_scale

        with torch.no_grad():
            self.bias.copy_(data_scaled_mean)
            self.scale.copy_(data_scale)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized.item():
            self.initialize(x)

        return x.mul(self.scale).sub(self.bias)


class PosLinear(torch.nn.Linear):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_features = x.size(-1)
        scale = 1.0 / math.sqrt(max(1, in_features))
        return torch.nn.functional.linear(
            x, torch.nn.functional.softplus(self.weight), self.bias
        ) * scale


class PICNN(torch.nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        output_dimension: int = 1,
    ):
        super(PICNN, self).__init__()

        self.number_of_hidden_layers = int(number_of_hidden_layers)

        x_dimension = feature_dimension
        y_dimension = response_dimension
        u_dimension, z_dimension = hidden_dimension, hidden_dimension

        self.z_activation = torch.nn.Softplus()
        self.z_activation_inverse_one = math.log(math.exp(1) - 1)

        self.u_activation = torch.nn.ELU()
        self.u_activation_inverse_one = 1.

        self.positive_activation = torch.nn.Softplus()
        self.positive_activation_inverse_one = math.log(math.exp(1) - 1)

        self.first_linear_layer_tilde = torch.nn.Linear(x_dimension, u_dimension)
        self.first_linear_layer_uy = torch.nn.Linear(x_dimension, y_dimension)
        self.first_linear_layer_y = torch.nn.Linear(
            y_dimension, z_dimension, bias=False
        )
        self.first_linear_layer_u = torch.nn.Linear(
            x_dimension, z_dimension, bias=False
        )
        self.first_layer_z_activation_normalization = ActNorm(
            feature_dimension=z_dimension
        )

        self.linear_layers_tilde = torch.nn.ModuleList(
            [
                torch.nn.Linear(u_dimension, u_dimension)
                for _ in range(number_of_hidden_layers)
            ]
        )
        self.linear_layers_uz = torch.nn.ModuleList(
            [
                torch.nn.Linear(u_dimension, z_dimension)
                for _ in range(number_of_hidden_layers)
            ]
        )
        self.linear_layers_z = torch.nn.ModuleList(
            [
                PosLinear(z_dimension, z_dimension)
                for _ in range(number_of_hidden_layers)
            ]
        )
        self.linear_layers_uy = torch.nn.ModuleList(
            [
                torch.nn.Linear(u_dimension, y_dimension)
                for _ in range(number_of_hidden_layers)
            ]
        )
        self.linear_layers_y = torch.nn.ModuleList(
            [
                torch.nn.Linear(y_dimension, z_dimension, bias=False)
                for _ in range(number_of_hidden_layers)
            ]
        )
        self.linear_layers_u = torch.nn.ModuleList(
            [
                torch.nn.Linear(u_dimension, z_dimension, bias=False)
                for _ in range(number_of_hidden_layers)
            ]
        )
        self.z_activation_normalization = torch.nn.ModuleList(
            [
                ActNorm(feature_dimension=z_dimension)
                for _ in range(number_of_hidden_layers)
            ]
        )

        self.last_linear_layer_uz = torch.nn.Linear(u_dimension, z_dimension)
        self.last_linear_layer_z = PosLinear(z_dimension, output_dimension)
        self.last_linear_layer_uy = torch.nn.Linear(u_dimension, y_dimension)
        self.last_linear_layer_y = torch.nn.Linear(
            y_dimension, output_dimension, bias=False
        )
        self.last_linear_layer_u = torch.nn.Linear(
            u_dimension, output_dimension, bias=False
        )

    def forward(self, x, y):
        u = self.u_activation(
            self.first_linear_layer_tilde(x) + self.u_activation_inverse_one
        )
        z = self.z_activation(
            self.first_layer_z_activation_normalization(
                self.first_linear_layer_y(y * self.first_linear_layer_uy(x)) +
                self.first_linear_layer_u(x)
            ) + self.z_activation_inverse_one
        )

        for iteration_number in range(self.number_of_hidden_layers):
            u, z = (
                self.u_activation(
                    self.linear_layers_tilde[iteration_number](u) +
                    self.u_activation_inverse_one
                ),
                self.z_activation(
                    self.z_activation_normalization[iteration_number](
                        self.linear_layers_z[iteration_number](
                            z * self.positive_activation(
                                (self.linear_layers_uz[iteration_number]
                                 (u)) + self.positive_activation_inverse_one
                            )
                        ) + self.linear_layers_y[iteration_number]
                        (y * self.linear_layers_uy[iteration_number](u)) +
                        self.linear_layers_u[iteration_number](u)
                    ) + self.z_activation_inverse_one
                )
            )

        output = self.last_linear_layer_z(
            z * self.positive_activation(
                self.last_linear_layer_uz(u)
                + self.positive_activation_inverse_one
            )
        ) + \
        self.last_linear_layer_y(
            y * self.last_linear_layer_uy(u)
        ) + \
        self.last_linear_layer_u(u)

        return self.z_activation(output)


class PISCNN(PICNN):

    def __init__(self, *args, **kwargs):
        super(PISCNN, self).__init__(*args, **kwargs)
        self.weight_for_convexity = torch.nn.Parameter(
            torch.log(torch.tensor(1e-1) * 0.5)
        )

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return self.weight_for_convexity.exp() * tensor.norm(dim=-1, keepdim=True
                                                             ).pow(2) + output


class FFNN(torch.nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        output_dimension: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.activation_function = torch.nn.Softplus()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(torch.nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.potential_network = torch.nn.Sequential(
            torch.nn.Linear(response_dimension + feature_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            torch.nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        input_tensor = torch.cat([condition, tensor], dim=-1)
        output_tensor = self.potential_network(input_tensor)
        return output_tensor


class FFNN(torch.nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        output_dimension: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.activation_function = torch.nn.Softplus()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(torch.nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.potential_network = torch.nn.Sequential(
            torch.nn.Linear(response_dimension + feature_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            torch.nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        assert condition.ndim == tensor.ndim, "condition and tensor must have the same number of dimensions"

        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
            tensor = tensor.unsqueeze(0)

        input_tensor = torch.cat([condition, tensor], dim=-1)
        output_tensor = self.potential_network(input_tensor)
        return output_tensor


class SCFFNN(FFNN):
    """Feed Forward Neural Network with added norm to enforce strong convexity.
    alpha: regularization parameter
    """

    def __init__(self, *args, **kwargs):
        super(SCFFNN, self).__init__(*args, **kwargs)
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return output + 0.5 * torch.exp(self.log_alpha
                                        ) * torch.norm(tensor, dim=-1, keepdim=True)**2


network_type_name_to_network_type: dict[str, torch.nn.Module] = {
    "SCFFNN": SCFFNN,
    "FFNN": FFNN,
    "PICNN": PICNN,
    "PISCNN": PISCNN
}
