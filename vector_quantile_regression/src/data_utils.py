import numpy as np
import matplotlib.pyplot as plt

def create_linear_joint_x_y(n_points: int):
    """
    Creating data in the linear form for x values distributed between 1 and 5.

    Args:
        n_points (int): The number of data points to generate.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): 1D array of x-coordinates, distributed between 1 and 5.
            - y (numpy.ndarray): 2D array of y-coordinates, derived from x and random noise.
    """
    u = np.random.normal(0, 1, size=(n_points, 2))
    x = np.random.uniform(0.5, 2.5, size=(n_points, 1))

    y = np.concatenate([
        u[:, 0:1] * x,
        u[:, 1:2] * x + x,
    ], axis=1)

    return x, y


def create_linear_conditional_x(n_points: int, x_value: float):
    """
    Creating data in linear form

    Args:
        n_points (int): The number of data points to generate.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): 1D array of x-coordinates, all equals to x.
            - y (numpy.ndarray): 2D array of y-coordinates, derived from x and random noise.
    """
    u = np.random.normal(0, 1, size=(n_points, 2))
    x = np.ones((n_points, 1)) * x_value

    y = np.concatenate([
        u[:, 0:1] * x,
        u[:, 1:2] * x + x,
    ], axis=1)

    return x, y

def create_joint_x_y(n_points: int):
    """
    Creating data in the form of a banana with x values distributed between 1 and 5.

    Args:
        n_points (int): The number of data points to generate.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): 1D array of x-coordinates, distributed between 1 and 5.
            - y (numpy.ndarray): 2D array of y-coordinates, derived from x and random noise.
    """
    u = np.random.normal(0, 1, size=(n_points, 2))
    x = np.random.uniform(0.5, 2.5, size=(n_points, 1))

    y = np.concatenate([
        u[:, 0:1] * x,
        u[:, 1:2] / x + (u[:, 0:1]**2 + x**3),
    ], axis=1)

    return x, y

def create_conditional_x(n_points: int, x_value: float):
    """
    Creating data in the form of a banana with fixed

    Args:
        n_points (int): The number of data points to generate.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): 1D array of x-coordinates, all equals to x.
            - y (numpy.ndarray): 2D array of y-coordinates, derived from x and random noise.
    """
    u = np.random.normal(0, 1, size=(n_points, 2))
    x = np.ones((n_points, 1)) * x_value

    y = np.concatenate([
        u[:, 0:1] * x_value,
        u[:, 1:2] / x_value + (u[:, 0:1]**2 + x_value**3),
    ], axis=1)

    return x, y

if __name__ == "__main__":
    num_points_to_generate = 1000
    x, y = create_joint_x_y(num_points_to_generate)

    print(f"Shape of x: {x.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Min x value: {np.min(x):.2f}, Max x value: {np.max(x):.2f}")


    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x[:,0 ], y[:, 0], y[:, 1], c='blue', marker='o', s=30, alpha=0.6)

    ax.set_xlabel('X-axis (distributed between 1 and 5)')
    ax.set_ylabel('Y1')
    ax.set_zlabel('Y2')

    ax.grid(True)
    ax.view_init(elev=20, azim=120)

    plt.show()
