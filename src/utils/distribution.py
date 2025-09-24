import torch

def sample_distribution(shape: tuple, distribution_name: str):
    if distribution_name == "uniform_centered":
        return sample_uniform_centered(shape)
    elif distribution_name == "uniform":
        return torch.rand(shape)
    elif distribution_name == "uniform_ball":
        return sample_uniform_ball(shape)
    elif distribution_name == "normal":
        return torch.randn(shape)
    elif distribution_name == "exponential":
        return sample_exponential(shape)
    elif distribution_name == "exponential-centered":
        return sample_exponential(shape) - 1
    else:
        raise ValueError(
            f"Distribution {distribution_name} not supported."
            " Supported distributions are:", "uniform_centered"
            ", uniform"
            ", uniform_ball"
            ", normal"
            ", exponential"
            ", exponential-centered"
        )

def sample_uniform_ball_surface(shape: tuple):
    point = torch.randn(*shape)
    point_normalized = point / point.norm(dim=-1, keepdim=True)
    return point_normalized

def sample_uniform_ball_surface_like(tensor: torch.Tensor):
    point = torch.randn_like(tensor)
    point_normalized = point / point.norm(dim=-1, keepdim=True)
    return point_normalized

def sample_exponential(shape: tuple):
    return -torch.log(1 - torch.rand(shape))

def sample_uniform_centered(shape: tuple):
    return torch.rand(shape) * 2 - 1

def sample_uniform_ball(shape: tuple):
    point = torch.randn(*shape[:-1], shape[-1] + 2)
    point_normalized = point / point.norm(dim=-1, keepdim=True)
    return point_normalized

def sample_distribution_like(tensor: torch.Tensor, distribution_name: str):
    if distribution_name == "uniform_centered":
        return uniform_centered_like(tensor)
    elif distribution_name == "uniform":
        return torch.rand_like(tensor)
    elif distribution_name == "uniform_ball":
        return uniform_ball_like(tensor)
    elif distribution_name == "normal":
        return torch.randn_like(tensor)
    elif distribution_name == "exponential":
        return exponential_like(tensor)
    elif distribution_name == "exponential-centered":
        return exponential_like(tensor) - 1
    else:
        raise ValueError(
            f"Distribution {distribution_name} not supported."
            " Supported distributions are:", "uniform_centered"
            ", uniform"
            ", uniform_ball"
            ", normal"
            ", exponential"
            ", exponential-centered"
        )

def exponential_like(tensor: torch.Tensor):
    return -torch.log(1 - torch.rand_like(tensor))

def uniform_centered_like(tensor: torch.Tensor):
    return torch.rand_like(tensor) * 2 - 1

def uniform_ball_like(tensor: torch.Tensor):
    point = torch.randn(*tensor.shape[:-1], tensor.shape[-1] + 2)
    point_normalized = point / point.norm(dim=-1, keepdim=True)
    return point_normalized[..., :-2]
