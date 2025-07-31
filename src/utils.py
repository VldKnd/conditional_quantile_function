import torch
from typing_extensions import TypedDict, Literal
import matplotlib.pyplot as plt
from old_source_code.data import create_conditional_x
import matplotlib
import scipy.stats as stats
from protocols.pushforward_operator import PushForwardOperator

class TrainParams(TypedDict):
    num_epochs: int | None = None
    learning_rate: float | None = None
    verbose: bool = False

def get_quantile_level_analytically(alpha: torch.Tensor, distribution: Literal["gaussian", "ball"], dimension: int) -> torch.Tensor:
    """Function finds the radius, that is corresponding to alpha-quantile of the samples.

    The function is based on the fact, that the distribution of the distances is symmetric around the origin.
    So, we can find the radius, that is corresponding to alpha-quantile of the samples.

    Args:
        samples (torch.Tensor): Samples from the distribution.
        alpha (float): Level of the quantile.

    Returns:
        float: The radius of the quantile level.
    """
    if distribution == "gaussian":
        scipy_quantile = stats.chi2.ppf(alpha.cpu().detach().numpy(), df=dimension)
        return torch.from_numpy(scipy_quantile**(1/2))
    elif distribution == "ball":
        return alpha**(1/dimension)
    else:
        raise ValueError(f"Distribution {distribution} is not supported.")

def get_quantile_level_numerically(samples: torch.Tensor, alpha: float) -> float:
    """Function finds the radius, that is corresponding to alpha-quantile of the samples.

    The function is based on the fact, that the distribution of the distances is symmetric around the origin.
    So, we can find the radius, that is corresponding to alpha-quantile of the samples.

    Args:
        samples (torch.Tensor): Samples from the distribution.
        alpha (float): Level of the quantile.

    Returns:
        float: The radius of the quantile level.
    """
    distances = torch.norm(samples, dim=-1).reshape(-1)
    distances, _ = distances.sort()
    return distances[int(alpha * len(distances))]

def plot_potentials_from_banana_dataset(model: PushForwardOperator, device_and_dtype_specifications: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Separated 3D Plots', fontsize=16)

    ax1.set_title('Conditional Scatter Data (y_x_gt)')
    ax1.set_xlabel('Axis 0')
    ax1.set_ylabel('Axis 1')
    ax1.set_zlabel('x_ value')

    for x_ in range(50, 250, 10):
        x = torch.tensor([x_ / 100, 1])

        # This section is now active for the first plot
        _, y_x_gt = create_conditional_x(n_points=100, x_value=x[0].item())
        z_scatter = torch.full((_.shape[0],),  x_)
        ax1.scatter(y_x_gt[:, 0], y_x_gt[:, 1], z_scatter, color='blue', marker='o', s=30, alpha=0.2)

    ax1.view_init(elev=-55, azim=154, roll=-83)

    ax2.set_title('Contour Lines')
    ax2.set_xlabel('Axis 0')
    ax2.set_ylabel('Axis 1')
    ax2.set_zlabel('x_ value')
    color_map = matplotlib.colormaps['viridis']

    loop_start_value = 50
    for x_ in range(loop_start_value, 250, 10):
        X_batch = torch.tensor([[x_ / 100]]).repeat(100, 1)
        quantile_levels = torch.arange(0.05, 1, 0.1)
        radii = get_quantile_level_analytically(quantile_levels, distribution="gaussian", dimension=2)

        colors = [color_map(i / len(radii)) for i in range(len(radii))]
        for i, contour_radius in enumerate(radii):
            color = colors[i]
            pi = torch.linspace(-torch.pi, torch.pi, 100)

            u = torch.stack([
                contour_radius * torch.cos(pi),
                contour_radius * torch.sin(pi),
            ]).T

            X_batch = X_batch.to(**device_and_dtype_specifications)
            u = u.to(**device_and_dtype_specifications)

            pushforward_of_u = model.push_forward_u_given_x(u, X=X_batch).detach().cpu()
            z_line = torch.full((pushforward_of_u.shape[0], ), x_)

            label = f'Quantile level {quantile_levels[i]:.2f}' if x_ == loop_start_value else ""
            ax2.plot(pushforward_of_u[:, 0], pushforward_of_u[:, 1], z_line, color=color, linewidth=2.5, label=label)
            ax1.plot(pushforward_of_u[:, 0], pushforward_of_u[:, 1], z_line, color=color, linewidth=2.5, label=label)

    ax2.view_init(elev=-55, azim=154, roll=-83)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def get_total_number_of_parameters(model: torch.nn.Module) -> int:
    """Get the total number of parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters())