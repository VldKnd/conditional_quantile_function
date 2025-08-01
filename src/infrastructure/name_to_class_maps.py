from datasets.protocol import Dataset
from datasets.synthetic.banana import BananaDataset
from datasets.synthetic.tictac import TicTacDataset
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators.quantile_regression.entropic_optimal_transport import EntropicOTQuantileRegression
from pushforward_operators.cpflow.core_flow import CPFlow
from pushforward_operators.quantile_regression.unconstrained_optimal_transport import UnconstrainedOTQuantileRegression
from pushforward_operators.quantile_regression.fast_non_linear_vector_quantile_regression import FastNonLinearVectorQuantileRegression
from pushforward_operators.quantile_regression.linear_quantile import LinearVectorQuantileRegression

name_to_dataset_map: dict[str, Dataset] = {
    "banana": BananaDataset,
    "tictac": TicTacDataset,
}

name_to_pushforward_operator_map: dict[str, PushForwardOperator] = {
    "entropic_optimal_transport_quantile_regression": EntropicOTQuantileRegression,
    "convex_potential_flow": CPFlow,
    "unconstrained_optimal_transport_quantile_regression": UnconstrainedOTQuantileRegression,
    "fast_non_linear_vector_quantile_regression": FastNonLinearVectorQuantileRegression,
    "linear_vector_quantile_regression": LinearVectorQuantileRegression,
}