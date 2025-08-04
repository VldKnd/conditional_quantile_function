import matplotlib.pyplot as plt
import torch
import matplotlib
from datasets import BananaDataset
from pushforward_operators import PushForwardOperator
from utils.quantile import get_quantile_level_analytically

def plot_potentials_from_banana_dataset(model: PushForwardOperator, number_of_conditional_points:int, number_of_points_to_sample:int, tensor_parameteres: dict):
    dataset = BananaDataset()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Separated 3D Plots', fontsize=16)

    ax1.set_title('Conditional Scatter Data (y_x_gt)')
    ax1.set_xlabel('Axis 0')
    ax1.set_ylabel('Axis 1')
    ax1.set_zlabel('x_ value')

    X = torch.linspace(1, 2.5, number_of_conditional_points).unsqueeze(1)

    # This section is now active for the first plot
    y_x_gt = dataset.sample_conditional(n_points=number_of_points_to_sample, X=X)
    z_scatter = X.unsqueeze(1).repeat(1, number_of_points_to_sample, 1)
    y_0 = y_x_gt[:, :, 0].flatten()
    y_1 = y_x_gt[:, :, 1].flatten()
    z_scatter = z_scatter.flatten()
    ax1.scatter(y_0, y_1, z_scatter, color='blue', marker='o', s=30, alpha=0.2)

    ax1.view_init(elev=-55, azim=154, roll=-83)

    ax2.set_title('Contour Lines')
    ax2.set_xlabel('Axis 0')
    ax2.set_ylabel('Axis 1')
    ax2.set_zlabel('x_ value')
    color_map = matplotlib.colormaps['viridis']
    quantile_levels = torch.arange(0.05, 1, 0.1)
    radii = get_quantile_level_analytically(quantile_levels, distribution="gaussian", dimension=2)
    colors = [color_map(i / len(radii)) for i in range(len(radii))]
    X = X.to(**tensor_parameteres)

    for j in range(number_of_conditional_points):
        X_batch = X[j].unsqueeze(0).repeat(100, 1)
        for i, contour_radius in enumerate(radii):
            color = colors[i]
            pi = torch.linspace(-torch.pi, torch.pi, 100)

            u = torch.stack([
                contour_radius * torch.cos(pi),
                contour_radius * torch.sin(pi),
            ]).T
            u = u.to(**tensor_parameteres)

            pushforward_of_u = model.push_forward_u_given_x(u, X=X_batch).detach().cpu()
            z_line = torch.full((pushforward_of_u.shape[0], ), X[j].item())

            label = f'Quantile level {quantile_levels[i]:.2f}' if j == 0 else ""
            ax2.plot(pushforward_of_u[:, 0], pushforward_of_u[:, 1], z_line, color=color, linewidth=2.5, label=label)
            ax1.plot(pushforward_of_u[:, 0], pushforward_of_u[:, 1], z_line, color=color, linewidth=2.5, label=label)

    ax2.view_init(elev=-55, azim=154, roll=-83)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_quantile_levels_from_banana_dataset(model: PushForwardOperator, number_of_points_to_sample:int, tensor_parameteres: dict, conditional_value: float, number_of_quantile_levels: int):
    dataset = BananaDataset()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Separated 3D Plots', fontsize=16)

    ax1.set_title('Conditional Scatter Data (y_x_gt)')
    ax1.set_xlabel('Axis 0')
    ax1.set_ylabel('Axis 1')
    ax1.set_zlabel('x_ value')

    X = torch.tensor([[conditional_value]])

    y_x_gt = dataset.sample_conditional(n_points=number_of_points_to_sample, X=X)
    z_scatter = X.unsqueeze(1).repeat(1, number_of_points_to_sample, 1)

    y_0 = y_x_gt[:, :, 0].flatten()
    y_1 = y_x_gt[:, :, 1].flatten()
    z_scatter = z_scatter.flatten()
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
    X = X.to(**tensor_parameteres)

    X_batch = X.repeat(100, 1)
    for i, contour_radius in enumerate(radii):
        color = colors[i]
        pi = torch.linspace(-torch.pi, torch.pi, 100)

        u = torch.stack([
            contour_radius * torch.cos(pi),
            contour_radius * torch.sin(pi),
        ]).T
        u = u.to(**tensor_parameteres)

        pushforward_of_u = model.push_forward_u_given_x(u, X=X_batch).detach().cpu()
        u_approximated = dataset.push_backward_Y_given_X(pushforward_of_u, X_batch)

        z_line = torch.full((pushforward_of_u.shape[0], ), X.item())

        label = f'Quantile level {quantile_levels[i]:.2f}'
        ax1.plot(pushforward_of_u[:, 0], pushforward_of_u[:, 1], z_line, color=color, linewidth=2.5, label=label)
        ax2.plot(u_approximated[:, 0], u_approximated[:, 1], z_line, color=color, linewidth=2.5, label=label)
        ax2.plot(u[:, 0], u[:, 1], z_line, ":", color=color, linewidth=2.5, label=label, )

    ax2.view_init(elev=-55, azim=154, roll=-83)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()