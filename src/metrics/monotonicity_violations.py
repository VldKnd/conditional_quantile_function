import torch
from datasets import QuadraticPotentialConvexBananaDataset, BananaDataset


def percentage_of_monotonicity_violation(
    tensor: torch.Tensor, pushforward_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Computes the percentage of co-monotonicity violation between two sets of points.
    """
    return (
        1 - tensor.unsqueeze(1).sub(tensor.unsqueeze(0)).mul(
            pushforward_tensor.unsqueeze(1).sub(pushforward_tensor.unsqueeze(0))
        ).sum(dim=-1).ge(0).float().mean()
    ) * 100


if __name__ == "__main__":
    n_points = 10000
    tensor = torch.randn(n_points, 2)
    dataset = QuadraticPotentialConvexBananaDataset(tensor_parameters={})
    tensor_pushforward = dataset.push_u_given_x(
        u=tensor, x=dataset.sample_covariates(1).repeat(n_points, 1)
    )
    print(percentage_of_monotonicity_violation(tensor, tensor_pushforward))

    tensor = torch.randn(n_points, 2)
    dataset = BananaDataset(tensor_parameters={})
    tensor_pushforward = dataset.push_u_given_x(
        u=tensor, x=dataset.sample_covariates(1).repeat(n_points, 1)
    )
    print(percentage_of_monotonicity_violation(tensor, tensor_pushforward))

    tensor = torch.randn(n_points, 2)
    tensor_pushforward = tensor**2
    print(percentage_of_monotonicity_violation(tensor, tensor_pushforward))
