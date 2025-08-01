from protocols.dataset import Dataset
from datasets.synthetic.banana import BananaDataset
from protocols.pushforward_operator import PushForwardOperator
from quantile_regression.entropic_optimal_transport import EntropicOTQuantileRegression
from cpflow.core_flow import CPFlow
from quantile_regression.unconstrained_optimal_transport import UnconstrainedOTQuantileRegression
from quantile_regression.fast_non_linear_vector_quantile_regression import FastNonLinearVectorQuantileRegression
from quantile_regression.linear_quantile import LinearVectorQuantileRegression

name_to_dataset_map: dict[str, Dataset] = {
    "banana": BananaDataset,
}

name_to_pushforward_operator_map: dict[str, PushForwardOperator] = {
    "entropic_optimal_transport_quantile_regression": EntropicOTQuantileRegression,
    "convex_potential_flow": CPFlow,
    "unconstrained_optimal_transport_quantile_regression": UnconstrainedOTQuantileRegression,
    "fast_non_linear_vector_quantile_regression": FastNonLinearVectorQuantileRegression,
    "linear_vector_quantile_regression": LinearVectorQuantileRegression,
}