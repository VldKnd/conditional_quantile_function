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


def plot_quantile_levels_from_dataset(
    model: PushForwardOperator, dataset: Dataset, conditional_value: torch.Tensor,
    number_of_quantile_levels: int, tensor_parameters: dict
):

    quantile_levels = torch.linspace(0.05, 0.95, number_of_quantile_levels)
    radii = get_quantile_level_analytically(
        quantile_levels, distribution="gaussian", dimension=2
    )

    X_batch = conditional_value.repeat(1000, 1).to(**tensor_parameters)
    list_of_ground_truth_U_quantiles = []
    list_of_approximated_U_quantiles = []

    list_of_ground_truth_Y_quantiles = []
    list_of_approximated_Y_quantiles = []

    with torch.no_grad():
        for _, contour_radius in enumerate(radii):
            pi = torch.linspace(-torch.pi, torch.pi, 1000)

            ground_truth_U_quantiles = (
                torch.stack(
                    [
                        contour_radius * torch.cos(pi),
                        contour_radius * torch.sin(pi),
                    ]
                ).T
            ).to(**tensor_parameters)
            ground_truth_Y_quantiles = dataset.push_u_given_x(
                u=ground_truth_U_quantiles, x=X_batch
            ).detach().cpu()

            try:
                approximated_U_quantiels = model.push_y_given_x(
                    y=ground_truth_Y_quantiles, x=X_batch
                )
            except NotImplementedError:
                approximated_U_quantiels = None

            try:
                approximated_Y_quantiels = model.push_u_given_x(
                    u=ground_truth_U_quantiles, x=X_batch
                )
            except NotImplementedError:
                approximated_Y_quantiels = None

            list_of_ground_truth_U_quantiles.append(ground_truth_U_quantiles)
            list_of_approximated_U_quantiles.append(approximated_U_quantiels)

            list_of_ground_truth_Y_quantiles.append(ground_truth_Y_quantiles)
            list_of_approximated_Y_quantiles.append(approximated_Y_quantiels)

    color_map = matplotlib.colormaps['viridis']
    fig, (ax1,
          ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})
    fig.suptitle('Separated 3D Plots', fontsize=16)

    ax1.set_title(f'Data space (Y|X={list(conditional_value.detach())})')
    ax1.set_xlabel('Axis 0')
    ax1.set_ylabel('Axis 1')
    ax1.set_zlabel('x_ value')

    z_line = torch.zeros(X_batch.shape[0]).flatten().detach()

    for i, (ground_truth_Y_quantiles, approximated_Y_quantiels) in enumerate(
        zip(list_of_ground_truth_Y_quantiles, list_of_approximated_Y_quantiles)
    ):
        color = color_map(i / number_of_quantile_levels)
        label = f'Quantile level {quantile_levels[i]:.2f}'

        if approximated_Y_quantiels is not None:
            ax1.plot(
                approximated_Y_quantiels[:, 0],
                approximated_Y_quantiels[:, 1],
                z_line,
                color=color,
                linewidth=2.5,
                label=label
            )

        ax1.plot(
            ground_truth_Y_quantiles[:, 0],
            ground_truth_Y_quantiles[:, 1],
            z_line,
            ":",
            color=color,
            linewidth=2.5,
            label=label
        )

    for i, (ground_truth_U_quantiles, approximated_U_quantiels) in enumerate(
        zip(list_of_ground_truth_U_quantiles, list_of_approximated_U_quantiles)
    ):
        color = color_map(i / number_of_quantile_levels)
        label = f'Quantile level {quantile_levels[i]:.2f}'
        if approximated_U_quantiels is not None:
            ax2.plot(
                approximated_U_quantiels[:, 0],
                approximated_U_quantiels[:, 1],
                z_line,
                color=color,
                linewidth=2.5,
                label=label
            )

        ax2.plot(
            ground_truth_U_quantiles[:, 0],
            ground_truth_U_quantiles[:, 1],
            z_line,
            ":",
            color=color,
            linewidth=2.5,
            label=label
        )

    ax1.view_init(elev=-55, azim=154, roll=-83)
    ax1.legend()

    ax2.view_init(elev=-55, azim=154, roll=-83)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
