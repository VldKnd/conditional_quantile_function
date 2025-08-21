import numpy as np
import torch

from pushforward_operators import PushForwardOperator
from datasets.protocol import Dataset
from utils.quantile import get_quantile_level_analytically

from metrics.wsc import wsc_unbiased


def sample_conditional_and_marginal_coverage(
        pushforward_operator: PushForwardOperator,
        X_dataset: torch.Tensor,
        Y_dataset: torch.Tensor,
        X_calibration: torch.Tensor,
        Y_calibration: torch.Tensor,
        number_of_quantile_levels: int = 20,
        verbose: bool = False):
    quantile_levels = torch.linspace(0.05, 0.95, number_of_quantile_levels)
    radii_true = get_quantile_level_analytically(
        quantile_levels, distribution="gaussian",
        dimension=Y_dataset.shape[1]).reshape(-1, 1)

    radii_predicted = pushforward_operator.push_y_given_x(
        y=Y_dataset, x=X_dataset).norm(dim=-1).reshape(1, -1).detach().cpu()

    radii_predicted_calibration = pushforward_operator.push_y_given_x(
        y=Y_calibration,
        x=X_calibration).norm(dim=-1).reshape(1, -1).detach().cpu()
    n_cal = Y_calibration.shape[0]
    conformal_threshold_per_level = torch.quantile(
        radii_predicted_calibration,
        (quantile_levels * (n_cal + 1) /
         n_cal).to(dtype=radii_predicted_calibration.dtype),
        dim=1,
        keepdim=False)

    if verbose:
        print(
            f"{quantile_levels.shape=}, {conformal_threshold_per_level.shape=}"
        )
        print(f"{radii_true.shape=}, {radii_predicted.shape=}")

    is_covered = torch.greater_equal(radii_true, radii_predicted)
    is_covered_conformal = torch.greater_equal(conformal_threshold_per_level,
                                               radii_predicted)

    if verbose:
        print(f"{is_covered.shape=}")
        print(f"{is_covered_conformal.shape=}")

    wsc_per_level = np.zeros(
        number_of_quantile_levels)  #, dtype=quantile_levels.dtype)
    wsc_per_level_cp = np.zeros(number_of_quantile_levels)
    for i, quantile_level in enumerate(quantile_levels):
        if verbose:
            print(f"Computing WSC for level = {quantile_level}")
        wsc_per_level[i] = wsc_unbiased(
            reprs=X_dataset,
            coverages=is_covered[i].numpy(force=True),
            delta=0.1, n_cpus=8)
        wsc_per_level_cp[i] = wsc_unbiased(
            reprs=X_dataset,
            coverages=is_covered_conformal[i].numpy(force=True),
            delta=0.1, n_cpus=8)

    coverage_per_level = is_covered.numpy(force=True).mean(axis=-1)
    coverage_per_level_cp = is_covered_conformal.numpy(force=True).mean(
        axis=-1)

    if verbose:
        np.set_printoptions(precision=2)
        print(
            f"Nominal level:            {quantile_levels.numpy(force=True)} \n"
            f"Empirical coverage:       {coverage_per_level} \n"
            f"Empirical coverage (CP):  {coverage_per_level_cp} \n"
            f"Worst-slab coverage:      {wsc_per_level} \n"
            f"Worst-slab coverage (CP): {wsc_per_level_cp}")

    return quantile_levels, coverage_per_level, wsc_per_level, coverage_per_level_cp, wsc_per_level_cp
