import numpy as np

def estimate_gradient_knn(u_samples, f_samples, point, k):
    """
    Estimates the gradient of a scalar function f: R^d -> R at a single
    given point using k-nearest neighbors and ordinary least-squares.

    This function fits a local linear model (a hyperplane) to the k-nearest
    neighbors of the target point. The gradient of that model is the
    estimated gradient.

    Args:
        u_samples (np.ndarray): An (N, d) array of N input samples in d dimensions.
        f_samples (np.ndarray): An (N,) array of N corresponding scalar output samples.
        point (tuple or np.ndarray): The d-dimensional point (u_1, ..., u_d) at
            which to estimate the gradient.
        k (int): The number of nearest neighbors to use for the estimation.

    Returns:
        np.ndarray: The estimated (d,) gradient vector.
    """
    f_samples = f_samples.flatten()

    if u_samples.shape[0] != f_samples.shape[0]:
        raise ValueError("u_samples and f_samples must have the same number of samples.")

    N, d = u_samples.shape
    point = np.asarray(point)

    if point.shape != (d,):
        raise ValueError(f"The 'point' must be a d-dimensional vector, but got shape {point.shape} for d={d}.")

    if k < d + 1:
        raise ValueError(f"k must be at least d+1 (which is {d+1}) to fit a hyperplane in R^{d}.")
    if k > N:
        raise ValueError(f"k ({k}) cannot be larger than the number of samples ({N}).")

    distances_sq = np.sum((u_samples - point)**2, axis=1)
    knn_indices = np.argsort(distances_sq)[:k]

    local_u_samples = u_samples[knn_indices]
    local_f_samples = f_samples[knn_indices]

    A_design = np.hstack([local_u_samples, np.ones((k, 1))])

    try:
        coeffs = np.linalg.pinv(A_design) @ local_f_samples
    except np.linalg.LinAlgError:
        return np.full(d, np.nan)

    gradient = coeffs[:d]
    return gradient


def estimate_gradients_for_points_knn(u_samples, f_samples, points_of_interest, k):
    """
    Estimates the gradient for multiple points of interest by calling
    estimate_gradient_knn for each point.

    Args:
        u_samples (np.ndarray): An (N, d) array of input samples.
        f_samples (np.ndarray): An (N,) array of scalar output samples.
        points_of_interest (np.ndarray): An (M, d) array of M points where
            the gradient is to be estimated.
        k (int): The number of nearest neighbors to use.

    Returns:
        np.ndarray: An (M, d) array containing the M estimated gradients.
    """
    if points_of_interest.ndim == 1:
        points_of_interest = points_of_interest.reshape(1, -1)

    all_gradients = np.array([
        estimate_gradient_knn(u_samples, f_samples, point, k)
        for point in points_of_interest
    ])

    return all_gradients