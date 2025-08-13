from datasets import (
    BananaDataset,
    TicTacDataset,
    Dataset,
    StarDataset,
)
from pushforward_operators import (
    PushForwardOperator,
    EntropicOTQuantileRegression,
    CPFlow,
    UnconstrainedOTQuantileRegression,
    UnconstrainedAmortizedOTQuantileRegression,
    PICNNEntropicOTQuantileRegression,
)

name_to_dataset_map: dict[str, Dataset] = {
    "banana": BananaDataset,
    "tictac": TicTacDataset,
    "star": StarDataset,
}

name_to_pushforward_operator_map: dict[str, PushForwardOperator] = {
    "entropic_optimal_transport_quantile_regression": EntropicOTQuantileRegression,
    "convex_potential_flow": CPFlow,
    "unconstrained_optimal_transport_quantile_regression": UnconstrainedOTQuantileRegression,
    "unconstrained_amortized_optimal_transport_quantile_regression": UnconstrainedAmortizedOTQuantileRegression,
    "picnn_entropic_optimal_transport_quantile_regression": PICNNEntropicOTQuantileRegression,
}