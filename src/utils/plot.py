import matplotlib.pyplot as plt
import torch
import matplotlib
from datasets.synthetic.banana import BananaDataset
from pushforward_operators.protocol import PushForwardOperator
from utils.quantile import get_quantile_level_analytically

def plot_potentials_from_banana_dataset(model: PushForwardOperator, device_and_dtype_specifications: dict):
    dataset = BananaDataset()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Separated 3D Plots', fontsize=16)

    ax1.set_title('Conditional Scatter Data (y_x_gt)')
    ax1.set_xlabel('Axis 0')
    ax1.set_ylabel('Axis 1')
    ax1.set_zlabel('x_ value')

    for x_ in range(50, 250, 10):
        x = torch.tensor([x_ / 100])

        # This section is now active for the first plot
        _, y_x_gt = dataset.sample_conditional(n_points=100, x=x)
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