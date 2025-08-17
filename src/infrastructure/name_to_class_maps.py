from datasets import (
    BananaDataset,
    TicTacDataset,
    Dataset,
    ConvexBananaDataset,
    StarDataset,
)
from pushforward_operators import (
    PushForwardOperator,
    EntropicOTQuantileRegression,
    CPFlow,
    UnconstrainedOTQuantileRegression,
    UnconstrainedAmortizedOTQuantileRegression,
)

name_to_dataset_map: dict[str, Dataset] = {
    "banana": BananaDataset,
    "tictac": TicTacDataset,
    "convex_banana":ConvexBananaDataset,
    "star": StarDataset,
}

name_to_pushforward_operator_map: dict[str, PushForwardOperator] = {
    "entropic_optimal_transport_quantile_regression": EntropicOTQuantileRegression, # Paragraph 2.4.3
    "convex_potential_flow": CPFlow, # Paragraph 2.4.5
    "unconstrained_optimal_transport_quantile_regression": UnconstrainedOTQuantileRegression, # Paragraph 2.4.4
    "unconstrained_amortized_optimal_transport_quantile_regression": UnconstrainedAmortizedOTQuantileRegression, # Amortized varient of Paragraph 2.4.4, https://arxiv.org/abs/2210.12153
}