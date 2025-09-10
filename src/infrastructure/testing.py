import torch
from tqdm import tqdm
from datasets import (
    # Prototype
    Dataset,
    # Synthetic datasets with pushforward operator
    BananaDataset,
    QuadraticPotentialConvexBananaDataset,
    NotConditionalBananaDataset,
    FNLVQR_MVN,
    PICNN_FNLVQR_Glasses,
    PICNN_FNLVQR_Star,
    PICNN_FNLVQR_Banana,
    #  Datasets that have only sample_joint implemented
    FNLVQR_Banana,
    FNLVQR_Star,
    FNLVQR_Glasses,
    FunnelDistribution,
)
from infrastructure.classes import Experiment
from infrastructure.name_to_class_maps import name_to_dataset_map, name_to_pushforward_operator_map
from metrics import (
    wassertein2, percentage_of_unexplained_variance, sliced_wasserstein2,
    kernel_density_estimate_kl_divergence, kernel_density_estimate_l1_divergence
)
from pushforward_operators import PushForwardOperator


def test_from_json_file(
    path_to_experiment_file: str,
    verbose: bool = False,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
) -> dict:
    experiment = Experiment.load_from_path_to_experiment_file(
        path_to_experiment_file=path_to_experiment_file
    )
    metrics = test(
        experiment,
        verbose=verbose,
        exclude_wasserstein2=exclude_wasserstein2,
        exclude_unexplained_variance_percentage=exclude_unexplained_variance_percentage,
        exclude_sliced_wasserstein2=exclude_sliced_wasserstein2,
        exclude_kde_kl_divergence=exclude_kde_kl_divergence,
        exclude_kde_l1_divergence=exclude_kde_l1_divergence
    )

    if experiment.path_to_metrics is not None:
        torch.save(metrics, experiment.path_to_metrics)

    return metrics


def load_pushforward_operator_from_experiment(
    experiment: Experiment
) -> PushForwardOperator:
    """
    Load a pushforward operator from an experiment.

    Args:
        experiment (Experiment): The experiment to load the pushforward operator from.

    Returns:
        PushForwardOperator: The loaded pushforward operator.
    """
    pushforward_operator = name_to_pushforward_operator_map[
        experiment.pushforward_operator_name](
            **experiment.pushforward_operator_parameters
        )
    pushforward_operator.to(**experiment.tensor_parameters)

    if experiment.path_to_weights is not None:
        pushforward_operator.load(
            experiment.path_to_weights,
            map_location=torch.device(experiment.tensor_parameters["device"])
        )
    else:
        raise ValueError("Path to the model is not specified. Model can not be loaded.")

    pushforward_operator.eval()
    return pushforward_operator


def test_on_dataset_with_defined_pushforward_operator(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    verbose: bool = False
) -> dict:
    """
    Test a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to test.
        exclude_wasserstein2 (bool): Whether to exclude the Wasserstein-2 distance.
        exclude_unexplained_variance_percentage (bool): Whether to exclude the unexplained variance percentage.
        exclude_sliced_wasserstein2 (bool): Whether to exclude the sliced Wasserstein-2 distance.
        exclude_kde_kl_divergence (bool): Whether to exclude the KDE KL divergence.
        exclude_kde_l1_divergence (bool): Whether to exclude the KDE L1 divergence.
        verbose (bool): Whether to print verbose output.
    """
    print("?")
    number_of_test_samples = 500
    number_of_generated_points = 2000

    dataset: Dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameters)

    metrics = {
        "Y_wasserstein2": [],
        "Y_sliced_wasserstein2": [],
        "Y_kde_kl_divergence": [],
        "Y_kde_l1_divergence": [],
        "Y|X_wasserstein2": [],
        "Y|X_sliced_wasserstein2": [],
        "Y|X_kde_kl_divergence": [],
        "Y|X_kde_l1_divergence": [],
        "YX_wasserstein2": [],
        "YX_sliced_wasserstein2": [],
        "YX_kde_kl_divergence": [],
        "YX_kde_l1_divergence": [],
        "U_wasserstein2": [],
        "U_sliced_wasserstein2": [],
        "U_kde_kl_divergence": [],
        "U_kde_l1_divergence": [],
        "U|X_wasserstein2": [],
        "U|X_sliced_wasserstein2": [],
        "U|X_kde_kl_divergence": [],
        "U|X_kde_l1_divergence": [],
        "UX_wasserstein2": [],
        "UX_sliced_wasserstein2": [],
        "UX_kde_kl_divergence": [],
        "UX_kde_l1_divergence": [],
        "Q^(-1)(Y,X)_uv_l2": [],
        "Q(U,X)_uv_l2": [],
    }

    random_number_generator = torch.Generator(
        device=experiment.tensor_parameters["device"]
    )
    random_number_generator.manual_seed(42)

    # Joint and Marginal
    for _ in tqdm(
        range(number_of_test_samples),
        desc="Running Marginal and Joint Tests",
        disable=not verbose
    ):
        X_tensor, Y_tensor, U_tensor = dataset.sample_x_y_u(
            n_points=number_of_generated_points
        )

        Y_approximation = pushforward_operator.push_u_given_x(U_tensor, X_tensor)
        U_approximation = pushforward_operator.push_y_given_x(Y_tensor, X_tensor)

        YX_tensor = torch.cat([Y_tensor, X_tensor], dim=1)
        YX_approximation = torch.cat([Y_approximation, X_tensor], dim=1)

        UX_tensor = torch.cat([U_tensor, X_tensor], dim=1)
        UX_approximation = torch.cat([U_approximation, X_tensor], dim=1)

        if not exclude_unexplained_variance_percentage:
            metrics["Q^(-1)(Y,X)_uv_l2"].append(
                percentage_of_unexplained_variance(Y_tensor, Y_approximation)
            )
            metrics["Q(U,X)_uv_l2"].append(
                percentage_of_unexplained_variance(U_tensor, U_approximation)
            )

        if not exclude_wasserstein2:
            metrics["Y_wasserstein2"].append(wassertein2(Y_tensor, Y_approximation))
            metrics["U_wasserstein2"].append(wassertein2(U_tensor, U_approximation))

            metrics["YX_wasserstein2"].append(wassertein2(YX_tensor, YX_approximation))
            metrics["UX_wasserstein2"].append(wassertein2(UX_tensor, UX_approximation))

        if not exclude_sliced_wasserstein2:
            metrics["Y_sliced_wasserstein2"].append(
                sliced_wasserstein2(Y_tensor, Y_approximation)
            )
            metrics["U_sliced_wasserstein2"].append(
                sliced_wasserstein2(U_tensor, U_approximation)
            )

            metrics["YX_sliced_wasserstein2"].append(
                sliced_wasserstein2(YX_tensor, YX_approximation)
            )
            metrics["UX_sliced_wasserstein2"].append(
                sliced_wasserstein2(UX_tensor, UX_approximation)
            )

        if not exclude_kde_kl_divergence or not exclude_kde_l1_divergence:
            X_sample, Y_sample = dataset.sample_joint(
                n_points=number_of_generated_points
            )
            U_sample = torch.randn_like(Y_sample)
            YX_sample = torch.cat([Y_sample, X_sample], dim=1)
            UX_sample = torch.cat([U_sample, X_sample], dim=1)

            if not exclude_kde_kl_divergence:
                metrics["Y_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

                metrics["YX_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        YX_tensor, YX_approximation, YX_sample
                    )
                )
                metrics["UX_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        UX_tensor, UX_approximation, UX_sample
                    )
                )

            if not exclude_kde_l1_divergence:
                metrics["Y_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )
                metrics["YX_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        YX_tensor, YX_approximation, YX_sample
                    )
                )
                metrics["UX_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        UX_tensor, UX_approximation, UX_sample
                    )
                )

    for _ in tqdm(
        range(number_of_test_samples),
        desc="Running Conditional Tests",
        disable=not verbose
    ):
        X_tensor = dataset.sample_covariates(1).repeat(number_of_generated_points, 1)
        U_tensor = torch.randn_like(Y_tensor)

        X_tensor, Y_tensor = dataset.sample_conditional(x=X_tensor)

        Y_approximation = pushforward_operator.push_u_given_x(U_tensor, X_tensor)
        U_approximation = pushforward_operator.push_y_given_x(Y_tensor, X_tensor)

        YX_tensor = torch.cat([Y_tensor, X_tensor], dim=1)
        YX_approximation = torch.cat([Y_approximation, X_tensor], dim=1)

        UX_tensor = torch.cat([U_tensor, X_tensor], dim=1)
        UX_approximation = torch.cat([U_approximation, X_tensor], dim=1)

        if not exclude_wasserstein2:
            metrics["Y|X_wasserstein2"].append(wassertein2(Y_tensor, Y_approximation))
            metrics["U|X_wasserstein2"].append(wassertein2(U_tensor, U_approximation))

        if not exclude_sliced_wasserstein2:
            metrics["Y|X_sliced_wasserstein2"].append(
                sliced_wasserstein2(Y_tensor, Y_approximation)
            )
            metrics["U|X_sliced_wasserstein2"].append(
                sliced_wasserstein2(U_tensor, U_approximation)
            )

        if not exclude_kde_kl_divergence or not exclude_kde_l1_divergence:
            X_sample, Y_sample = dataset.sample_conditional(x=X_tensor)
            U_sample = torch.randn_like(Y_sample)
            YX_sample = torch.cat([Y_sample, X_sample], dim=1)
            UX_sample = torch.cat([U_sample, X_sample], dim=1)

            if not exclude_kde_kl_divergence:
                metrics["Y|X_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U|X_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

            if not exclude_kde_l1_divergence:
                metrics["Y|X_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U|X_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

    return metrics


def test_on_dataset_with_defined_sample_joint(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
    verbose: bool = False
) -> dict:
    """
    Test a model on a synthetic dataset.

    Args:
        experiment (Experiment): The experiment to test.
        exclude_wasserstein2 (bool): Whether to exclude the Wasserstein-2 distance.
        exclude_unexplained_variance_percentage (bool): Whether to exclude the unexplained variance percentage.
        exclude_sliced_wasserstein2 (bool): Whether to exclude the sliced Wasserstein-2 distance.
        exclude_kde_kl_divergence (bool): Whether to exclude the KDE KL divergence.
        exclude_kde_l1_divergence (bool): Whether to exclude the KDE L1 divergence.
        verbose (bool): Whether to print verbose output.
    """
    number_of_test_samples = 500
    number_of_generated_points = 2000

    dataset: Dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )
    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameters)

    metrics = {
        "Y_wasserstein2": [],
        "Y_sliced_wasserstein2": [],
        "Y_kde_kl_divergence": [],
        "Y_kde_l1_divergence": [],
        "Y|X_wasserstein2": [],
        "Y|X_sliced_wasserstein2": [],
        "Y|X_kde_kl_divergence": [],
        "Y|X_kde_l1_divergence": [],
        "YX_wasserstein2": [],
        "YX_sliced_wasserstein2": [],
        "YX_kde_kl_divergence": [],
        "YX_kde_l1_divergence": [],
        "U_wasserstein2": [],
        "U_sliced_wasserstein2": [],
        "U_kde_kl_divergence": [],
        "U_kde_l1_divergence": [],
        "U|X_wasserstein2": [],
        "U|X_sliced_wasserstein2": [],
        "U|X_kde_kl_divergence": [],
        "U|X_kde_l1_divergence": [],
        "UX_wasserstein2": [],
        "UX_sliced_wasserstein2": [],
        "UX_kde_kl_divergence": [],
        "UX_kde_l1_divergence": [],
    }

    random_number_generator = torch.Generator(
        device=experiment.tensor_parameters["device"]
    )
    random_number_generator.manual_seed(42)

    # Joint and Marginal
    for _ in tqdm(
        range(number_of_test_samples),
        desc="Running Marginal and Joint Tests",
        disable=not verbose
    ):
        X_tensor, Y_tensor = dataset.sample_joint(n_points=number_of_generated_points)
        U_tensor = torch.randn_like(Y_tensor)

        Y_approximation = pushforward_operator.push_u_given_x(U_tensor, X_tensor)
        U_approximation = pushforward_operator.push_y_given_x(Y_tensor, X_tensor)

        YX_tensor = torch.cat([Y_tensor, X_tensor], dim=1)
        YX_approximation = torch.cat([Y_approximation, X_tensor], dim=1)

        UX_tensor = torch.cat([U_tensor, X_tensor], dim=1)
        UX_approximation = torch.cat([U_approximation, X_tensor], dim=1)

        if not exclude_wasserstein2:
            metrics["Y_wasserstein2"].append(wassertein2(Y_tensor, Y_approximation))
            metrics["U_wasserstein2"].append(wassertein2(U_tensor, U_approximation))

            metrics["YX_wasserstein2"].append(wassertein2(YX_tensor, YX_approximation))
            metrics["UX_wasserstein2"].append(wassertein2(UX_tensor, UX_approximation))

        if not exclude_sliced_wasserstein2:
            metrics["Y_sliced_wasserstein2"].append(
                sliced_wasserstein2(Y_tensor, Y_approximation)
            )
            metrics["U_sliced_wasserstein2"].append(
                sliced_wasserstein2(U_tensor, U_approximation)
            )

            metrics["YX_sliced_wasserstein2"].append(
                sliced_wasserstein2(YX_tensor, YX_approximation)
            )
            metrics["UX_sliced_wasserstein2"].append(
                sliced_wasserstein2(UX_tensor, UX_approximation)
            )

        if not exclude_kde_kl_divergence or not exclude_kde_l1_divergence:
            X_sample, Y_sample = dataset.sample_joint(
                n_points=number_of_generated_points
            )
            U_sample = torch.randn_like(Y_sample)
            YX_sample = torch.cat([Y_sample, X_sample], dim=1)
            UX_sample = torch.cat([U_sample, X_sample], dim=1)

            if not exclude_kde_kl_divergence:
                metrics["Y_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

                metrics["YX_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        YX_tensor, YX_approximation, YX_sample
                    )
                )
                metrics["UX_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        UX_tensor, UX_approximation, UX_sample
                    )
                )

            if not exclude_kde_l1_divergence:
                metrics["Y_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )
                metrics["YX_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        YX_tensor, YX_approximation, YX_sample
                    )
                )
                metrics["UX_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        UX_tensor, UX_approximation, UX_sample
                    )
                )

    for _ in tqdm(
        range(number_of_test_samples),
        desc="Running Conditional Tests",
        disable=not verbose
    ):
        X_tensor = dataset.sample_covariates(1).repeat(number_of_generated_points, 1)
        X_tensor, Y_tensor = dataset.sample_conditional(x=X_tensor)
        U_tensor = torch.randn_like(Y_tensor)

        Y_approximation = pushforward_operator.push_u_given_x(U_tensor, X_tensor)
        U_approximation = pushforward_operator.push_y_given_x(Y_tensor, X_tensor)

        YX_tensor = torch.cat([Y_tensor, X_tensor], dim=1)
        YX_approximation = torch.cat([Y_approximation, X_tensor], dim=1)

        UX_tensor = torch.cat([U_tensor, X_tensor], dim=1)
        UX_approximation = torch.cat([U_approximation, X_tensor], dim=1)

        if not exclude_wasserstein2:
            metrics["Y|X_wasserstein2"].append(wassertein2(Y_tensor, Y_approximation))
            metrics["U|X_wasserstein2"].append(wassertein2(U_tensor, U_approximation))

        if not exclude_sliced_wasserstein2:
            metrics["Y|X_sliced_wasserstein2"].append(
                sliced_wasserstein2(Y_tensor, Y_approximation)
            )
            metrics["U|X_sliced_wasserstein2"].append(
                sliced_wasserstein2(U_tensor, U_approximation)
            )

        if not exclude_kde_kl_divergence or not exclude_kde_l1_divergence:
            X_sample, Y_sample = dataset.sample_conditional(x=X_tensor)
            U_sample = torch.randn_like(Y_sample)
            YX_sample = torch.cat([Y_sample, X_sample], dim=1)
            UX_sample = torch.cat([U_sample, X_sample], dim=1)

            if not exclude_kde_kl_divergence:
                metrics["Y|X_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U|X_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

            if not exclude_kde_l1_divergence:
                metrics["Y|X_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        Y_tensor, Y_approximation, Y_sample
                    )
                )
                metrics["U|X_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        U_tensor, U_approximation, U_sample
                    )
                )

    return metrics


def test(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
    verbose: bool = False
) -> dict:
    dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )

    if type(dataset) in {
        BananaDataset, QuadraticPotentialConvexBananaDataset,
        NotConditionalBananaDataset, FNLVQR_MVN, PICNN_FNLVQR_Glasses,
        PICNN_FNLVQR_Star, PICNN_FNLVQR_Banana
    }:
        return test_on_dataset_with_defined_pushforward_operator(
            experiment, exclude_wasserstein2, exclude_unexplained_variance_percentage,
            exclude_sliced_wasserstein2, exclude_kde_kl_divergence,
            exclude_kde_l1_divergence, verbose
        )
    elif type(dataset) in {
        FNLVQR_Banana, FNLVQR_Star, FNLVQR_Glasses, FunnelDistribution
    }:
        return test_on_dataset_with_defined_sample_joint(
            experiment, exclude_wasserstein2, exclude_sliced_wasserstein2,
            exclude_kde_kl_divergence, exclude_kde_l1_divergence, verbose
        )
    else:
        raise NotImplementedError(
            f"Testing on the dataset {dataset.__class__.__name__} is not implemented."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_experiment_file", type=str, required=True)
    parser.add_argument("--verbose", type=bool, required=False, default=True)
    parser.add_argument(
        "--exclude-sliced-wasserstein2",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-wasserstein2", action="store_true", required=False, default=False
    )
    parser.add_argument(
        "--exclude-unexplained-variance",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-kde-kl-divergence",
        action="store_true",
        required=False,
        default=False
    )
    parser.add_argument(
        "--exclude-kde-l1-divergence",
        action="store_true",
        required=False,
        default=False
    )

    args = parser.parse_args()

    test_from_json_file(
        args.path_to_experiment_file, args.verbose, args.exclude_wasserstein2,
        args.exclude_unexplained_variance, args.exclude_sliced_wasserstein2,
        args.exclude_kde_kl_divergence, args.exclude_kde_l1_divergence
    )
