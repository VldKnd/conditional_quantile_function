import torch
from torch import nn
from pushforward_operators.convex_potential_flow.icnn import PICNN as _PICNN


class PICNN(nn.Module):

    def __init__(
        self,
        feature_dimension: int,
        response_dimension: int,
        hidden_dimension: int,
        number_of_hidden_layers: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.picnn = _PICNN(
            dim=response_dimension,
            dimh=hidden_dimension,
            dimc=feature_dimension,
            num_hidden_layers=number_of_hidden_layers,
            symm_act_first=True,
            softplus_type="softplus",
            zero_softplus=True,
        )

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        assert condition.ndim == tensor.ndim
        if tensor.ndim == 1:
            tensor, condition = tensor.unsqueeze(0), condition.unsqueeze(0)
        return self.picnn(tensor, condition)


class PISCNN(PICNN):
    """Strongly convex variant of PICNN.
    alpha: regularization parameter
    """

    def __init__(self, *args, **kwargs):
        super(PISCNN, self).__init__(*args, **kwargs)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(1e-1)))

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return output + 0.5 * torch.exp(self.log_alpha
                                        ) * torch.norm(tensor, dim=-1, keepdim=True)**2


class FFNN(nn.Module):

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
        self.activation_function = nn.Softplus()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.potential_network = nn.Sequential(
            nn.Linear(response_dimension + feature_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            nn.Linear(hidden_dimension, output_dimension)
        )

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        input_tensor = torch.cat([condition, tensor], dim=-1)
        output_tensor = self.potential_network(input_tensor)
        return output_tensor


class PISCNN(PICNN):
    """Strongly convex variant of PICNN.
    alpha: regularization parameter
    """

    def __init__(self, *args, **kwargs):
        super(PISCNN, self).__init__(*args, **kwargs)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(1e-1)))

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return output + 0.5 * torch.exp(self.log_alpha
                                        ) * torch.norm(tensor, dim=-1, keepdim=True)**2


class FFNN(nn.Module):

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
        self.activation_function = nn.Softplus()

        hidden_layers = []
        for _ in range(number_of_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_dimension, hidden_dimension))
            hidden_layers.append(self.activation_function)

        self.potential_network = nn.Sequential(
            nn.Linear(response_dimension + feature_dimension, hidden_dimension),
            self.activation_function, *hidden_layers,
            nn.Linear(hidden_dimension, output_dimension)
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
        self.log_alpha = nn.Parameter(torch.tensor(0.))

    def forward(self, condition: torch.Tensor, tensor: torch.Tensor):
        output = super().forward(condition, tensor)
        return output + 0.5 * torch.exp(self.log_alpha
                                        ) * torch.norm(tensor, dim=-1, keepdim=True)**2


network_type_name_to_network_type: dict[str, nn.Module] = {
    "SCFFNN": SCFFNN,
    "FFNN": FFNN,
    "PICNN": PICNN,
    "PISCNN": PISCNN
}
