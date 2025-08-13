import matplotlib.pyplot as plt
import torch
import matplotlib
from datasets import Dataset
from pushforward_operators import PushForwardOperator
from utils.quantile import get_quantile_level_analytically

import matplotlib.pyplot as plt
import torch
import matplotlib
from pushforward_operators import PushForwardOperator
from utils.quantile import get_quantile_level_analytically

def plot_quantile_levels_from_dataset(model: PushForwardOperator, dataset: Dataset, conditional_value: torch.Tensor, number_of_quantile_levels: int, tensor_parameters: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Separated 3D Plots', fontsize=16)

    ax1.set_title('Conditional Scatter Data (y_x_gt)')
    ax1.set_xlabel('Axis 0')
    ax1.set_ylabel('Axis 1')
    ax1.set_zlabel('x_ value')

    X = conditional_value

    y_x_gt = dataset.sample_conditional(n_points=100, X=X)

    y_0 = y_x_gt[:, :, 0].flatten()
    y_1 = y_x_gt[:, :, 1].flatten()
    z_scatter = torch.zeros_like(y_0)
    ax1.scatter(y_0, y_1, z_scatter, color='blue', marker='o', s=30, alpha=0.2)

    ax1.view_init(elev=-55, azim=154, roll=-83)

    ax2.set_title('Contour Lines')
    ax2.set_xlabel('Axis 0')
    ax2.set_ylabel('Axis 1')
    ax2.set_zlabel('x_ value')
    color_map = matplotlib.colormaps['viridis']
    quantile_levels = torch.linspace(0.05, 0.95, number_of_quantile_levels)
    radii = get_quantile_level_analytically(quantile_levels, distribution="gaussian", dimension=2)
    colors = [color_map(i / len(radii)) for i in range(len(radii))]
    X = X.to(**tensor_parameters)

    X_batch = X.repeat(1000, 1)
    for i, contour_radius in enumerate(radii):
        color = colors[i]
        pi = torch.linspace(-torch.pi, torch.pi, 1000)

        u = torch.stack([
            contour_radius * torch.cos(pi),
            contour_radius * torch.sin(pi),
        ]).T
        u = u.to(**tensor_parameters)

        Y_quantile = dataset.push_u_given_x(u=u, x=X_batch).detach().cpu()
        u_approximated = model.push_y_given_x(y=Y_quantile, x=X_batch)

        z_line = torch.full((Y_quantile.shape[0], ), 0.)

        label = f'Quantile level {quantile_levels[i]:.2f}'
        ax1.plot(Y_quantile[:, 0], Y_quantile[:, 1], z_line, color=color, linewidth=2.5, label=label)
        ax2.plot(u_approximated[:, 0], u_approximated[:, 1], z_line, color=color, linewidth=2.5, label=label)
        ax2.plot(u[:, 0], u[:, 1], z_line, ":", color=color, linewidth=2.5, label=label, )

    ax2.view_init(elev=-55, azim=154, roll=-83)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()