import torch

def sample_sphere_surface(num_samples: int, dimension: int, **kwargs) -> torch.Tensor:
    """Generate n points on the surface of the d-dimensional sphere.

    Args:
        num_samples (int): The number of samples to generate.
        dimension (int): The dimension of the sphere.

    Returns:
        torch.Tensor: The samples.
    """
    vector = torch.randn(num_samples, dimension, **kwargs)
    distances = (vector**2).sum(dim=-1, keepdim=True)**0.5
    return vector / distances

def sample_ball(num_samples: int, dimension: int, **kwargs) -> torch.Tensor:
    """Generate n points inside the d-dimensional sphere.

    Args:
        num_samples (int): The number of samples to generate.
        dimension (int): The dimension of the sphere.

    Returns:
        torch.Tensor: The samples.
    """
    random_vectors = torch.randn(num_samples, dimension, **kwargs)
    vectors_norms = torch.norm(random_vectors, dim=1, keepdim=True)
    radius = torch.pow(torch.rand(num_samples, 1, **kwargs), 1. / dimension)
    return radius * random_vectors / vectors_norms