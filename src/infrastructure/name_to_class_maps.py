from datasets import (
    BananaDataset,
    TicTacDataset,
    Dataset
)
from pushforward_operators import (
    PushForwardOperator,
    EntropicOTQuantileRegression,
    CPFlow,
    UnconstrainedOTQuantileRegression,
    FastNonLinearVectorQuantileRegression,
    LinearVectorQuantileRegression,
)

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