import torch
from tqdm import trange
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

NUMBER_OF_JOINT_TEST_REPETITIONS = 100
NUMBER_OF_JOINT_TEST_SAMPLES = 2000

NUMBER_OF_CONDITIONAL_TEST_REPETITIONS = 100
NUMBER_OF_CONDITIONAL_TEST_CONDITIONS = 100
NUMBER_OF_CONDITIONAL_TEST_SAMPLES = 500


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


def test(
    experiment: Experiment,
    exclude_wasserstein2: bool = False,
    exclude_unexplained_variance_percentage: bool = False,
    exclude_sliced_wasserstein2: bool = False,
    exclude_kde_kl_divergence: bool = False,
    exclude_kde_l1_divergence: bool = False,
    verbose: bool = False,
) -> dict:
    dataset: Dataset = name_to_dataset_map[experiment.dataset_name](
        **experiment.dataset_parameters, tensor_parameters=experiment.tensor_parameters
    )

    if type(dataset) in {
        BananaDataset, QuadraticPotentialConvexBananaDataset,
        NotConditionalBananaDataset, FNLVQR_MVN, PICNN_FNLVQR_Glasses,
        PICNN_FNLVQR_Star, PICNN_FNLVQR_Banana
    }:
        pushforward_operator_not_defined = False
    elif type(dataset) in {
        FNLVQR_Banana, FNLVQR_Star, FNLVQR_Glasses, FunnelDistribution
    }:
        pushforward_operator_not_defined = True
    else:
        raise NotImplementedError(
            f"Testing on the dataset {dataset.__class__.__name__} is not implemented."
        )

    exclude_unexplained_variance_percentage = exclude_unexplained_variance_percentage or pushforward_operator_not_defined
    latent_distribution = experiment.latent_distribution_for_testing

    pushforward_operator = load_pushforward_operator_from_experiment(experiment)
    pushforward_operator.to(**experiment.tensor_parameters)
    pushforward_operator.eval()

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

    conditional_tests_progress_bar = trange(
        NUMBER_OF_CONDITIONAL_TEST_REPETITIONS,
        desc="Running Conditional Tests",
        disable=not verbose
    )

    for conditional_run_idx in conditional_tests_progress_bar:
        conditional_metrics = {
            "Y|X_wasserstein2": [],
            "Y|X_sliced_wasserstein2": [],
            "Y|X_kde_kl_divergence": [],
            "Y|X_kde_l1_divergence": [],
            "U|X_wasserstein2": [],
            "U|X_sliced_wasserstein2": [],
            "U|X_kde_kl_divergence": [],
            "U|X_kde_l1_divergence": [],
        }

        for conditional_sub_run_idx in range(NUMBER_OF_CONDITIONAL_TEST_CONDITIONS):
            covariates_tensor = dataset.sample_covariates(1).repeat(
                NUMBER_OF_CONDITIONAL_TEST_SAMPLES, 1
            )

            X_conditional_tensor, Y_conditional_tensor = dataset.sample_conditional(
                x=covariates_tensor
            )

            if latent_distribution == "gaussian":
                U_conditional_tensor = torch.randn_like(Y_conditional_tensor)
            elif latent_distribution == "uniform":
                U_conditional_tensor = torch.rand_like(Y_conditional_tensor)

            Y_conditional_approximation = pushforward_operator.push_u_given_x(
                U_conditional_tensor, X_conditional_tensor
            )
            U_conditional_approximation = pushforward_operator.push_y_given_x(
                Y_conditional_tensor, X_conditional_tensor
            )

            if not exclude_wasserstein2:
                conditional_metrics["Y|X_wasserstein2"].append(
                    wassertein2(Y_conditional_tensor, Y_conditional_approximation)
                )
                conditional_metrics["U|X_wasserstein2"].append(
                    wassertein2(U_conditional_tensor, U_conditional_approximation)
                )

            if not exclude_sliced_wasserstein2:
                conditional_metrics["Y|X_sliced_wasserstein2"].append(
                    sliced_wasserstein2(
                        Y_conditional_tensor, Y_conditional_approximation
                    )
                )
                conditional_metrics["U|X_sliced_wasserstein2"].append(
                    sliced_wasserstein2(
                        U_conditional_tensor, U_conditional_approximation
                    )
                )

            if not exclude_kde_kl_divergence:
                conditional_metrics["Y|X_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        Y_conditional_tensor, Y_conditional_approximation
                    )
                )
                conditional_metrics["U|X_kde_kl_divergence"].append(
                    kernel_density_estimate_kl_divergence(
                        U_conditional_tensor, U_conditional_approximation
                    )
                )

            if not exclude_kde_l1_divergence:
                conditional_metrics["Y|X_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        Y_conditional_tensor, Y_conditional_approximation
                    )
                )
                conditional_metrics["U|X_kde_l1_divergence"].append(
                    kernel_density_estimate_l1_divergence(
                        U_conditional_tensor, U_conditional_approximation
                    )
                )

            conditional_tests_progress_bar.set_postfix(
                {
                    "Conditional sub-run index": conditional_sub_run_idx,
                    "Conditional run index": conditional_run_idx,
                }
            )

        if not exclude_wasserstein2:
            metrics["Y|X_wasserstein2"].append(
                torch.stack(conditional_metrics["Y|X_wasserstein2"]).mean()
            )
            metrics["U|X_wasserstein2"].append(
                torch.stack(conditional_metrics["U|X_wasserstein2"]).mean()
            )

        if not exclude_sliced_wasserstein2:
            metrics["Y|X_sliced_wasserstein2"].append(
                torch.stack(conditional_metrics["Y|X_sliced_wasserstein2"]).mean()
            )
            metrics["U|X_sliced_wasserstein2"].append(
                torch.stack(conditional_metrics["U|X_sliced_wasserstein2"]).mean()
            )

        if not exclude_kde_kl_divergence:
            metrics["Y|X_kde_kl_divergence"].append(
                torch.stack(conditional_metrics["Y|X_kde_kl_divergence"]).mean()
            )
            metrics["U|X_kde_kl_divergence"].append(
                torch.stack(conditional_metrics["U|X_kde_kl_divergence"]).mean()
            )

        if not exclude_kde_l1_divergence:
            metrics["Y|X_kde_l1_divergence"].append(
                torch.stack(conditional_metrics["Y|X_kde_l1_divergence"]).mean()
            )
            metrics["U|X_kde_l1_divergence"].append(
                torch.stack(conditional_metrics["U|X_kde_l1_divergence"]).mean()
            )

    joint_tests_progress_bar = trange(
        NUMBER_OF_JOINT_TEST_REPETITIONS,
        desc="Running Marginal and Joint Tests",
        disable=not verbose
    )

    for joint_run_idx in joint_tests_progress_bar:
        X_joint_tensor, Y_joint_tensor = dataset.sample_joint(
            n_points=NUMBER_OF_JOINT_TEST_SAMPLES
        )

        if latent_distribution == "gaussian":
            U_joint_tensor = torch.randn_like(Y_joint_tensor)
        elif latent_distribution == "uniform":
            U_joint_tensor = torch.rand_like(Y_joint_tensor)

        Y_joint_approximation = pushforward_operator.push_u_given_x(
            U_joint_tensor, X_joint_tensor
        )
        U_joint_approximation = pushforward_operator.push_y_given_x(
            Y_joint_tensor, X_joint_tensor
        )

        YX_joint_tensor = torch.cat([Y_joint_tensor, X_joint_tensor], dim=1)
        YX_joint_approximation = torch.cat(
            [Y_joint_approximation, X_joint_tensor], dim=1
        )

        UX_joint_tensor = torch.cat([U_joint_tensor, X_joint_tensor], dim=1)
        UX_joint_approximation = torch.cat(
            [U_joint_approximation, X_joint_tensor], dim=1
        )

        if not exclude_unexplained_variance_percentage:
            X_joint_dataset, U_joint_dataset, Y_joint_dataset = dataset.sample_x_y_u(
                n_points=NUMBER_OF_JOINT_TEST_SAMPLES
            )

            U_joint_dataset_approximation = pushforward_operator.push_y_given_x(
                Y_joint_dataset, X_joint_dataset
            )
            Y_joint_dataset_approximation = pushforward_operator.push_u_given_x(
                U_joint_dataset, X_joint_dataset
            )

            metrics["Q^(-1)(Y,X)_uv_l2"].append(
                percentage_of_unexplained_variance(
                    Y_joint_dataset, Y_joint_dataset_approximation
                )
            )
            metrics["Q(U,X)_uv_l2"].append(
                percentage_of_unexplained_variance(
                    U_joint_dataset, U_joint_dataset_approximation
                )
            )

        if not exclude_wasserstein2:
            metrics["Y_wasserstein2"].append(
                wassertein2(Y_joint_tensor, Y_joint_approximation)
            )
            metrics["U_wasserstein2"].append(
                wassertein2(U_joint_tensor, U_joint_approximation)
            )

            metrics["YX_wasserstein2"].append(
                wassertein2(YX_joint_tensor, YX_joint_approximation)
            )
            metrics["UX_wasserstein2"].append(
                wassertein2(UX_joint_tensor, UX_joint_approximation)
            )

        if not exclude_sliced_wasserstein2:
            metrics["Y_sliced_wasserstein2"].append(
                sliced_wasserstein2(Y_joint_tensor, Y_joint_approximation)
            )
            metrics["U_sliced_wasserstein2"].append(
                sliced_wasserstein2(U_joint_tensor, U_joint_approximation)
            )

            metrics["YX_sliced_wasserstein2"].append(
                sliced_wasserstein2(YX_joint_tensor, YX_joint_approximation)
            )
            metrics["UX_sliced_wasserstein2"].append(
                sliced_wasserstein2(UX_joint_tensor, UX_joint_approximation)
            )

        if not exclude_kde_kl_divergence:
            metrics["Y_kde_kl_divergence"].append(
                kernel_density_estimate_kl_divergence(
                    Y_joint_tensor, Y_joint_approximation
                )
            )
            metrics["U_kde_kl_divergence"].append(
                kernel_density_estimate_kl_divergence(
                    U_joint_tensor, U_joint_approximation
                )
            )

            metrics["YX_kde_kl_divergence"].append(
                kernel_density_estimate_kl_divergence(
                    YX_joint_tensor, YX_joint_approximation
                )
            )
            metrics["UX_kde_kl_divergence"].append(
                kernel_density_estimate_kl_divergence(
                    UX_joint_tensor, UX_joint_approximation
                )
            )

        if not exclude_kde_l1_divergence:
            metrics["Y_kde_l1_divergence"].append(
                kernel_density_estimate_l1_divergence(
                    Y_joint_tensor, Y_joint_approximation
                )
            )
            metrics["U_kde_l1_divergence"].append(
                kernel_density_estimate_l1_divergence(
                    U_joint_tensor, U_joint_approximation
                )
            )
            metrics["YX_kde_l1_divergence"].append(
                kernel_density_estimate_l1_divergence(
                    YX_joint_tensor, YX_joint_approximation
                )
            )
            metrics["UX_kde_l1_divergence"].append(
                kernel_density_estimate_l1_divergence(
                    UX_joint_tensor, UX_joint_approximation
                )
            )

        joint_tests_progress_bar.set_postfix({"Joint run index": joint_run_idx})

    return metrics


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
