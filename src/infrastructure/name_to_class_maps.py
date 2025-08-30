from datasets import (
    BananaDataset,
    TicTacDataset,
    Dataset,
    ConvexBananaDataset,
    FNLVQR_MVN,
    NotConditionalBananaDataset,
)
from pushforward_operators import (
    PushForwardOperator,
    EntropicOTQuantileRegression,
    CPFlow,
    UnconstrainedOTQuantileRegression,
    UnconstrainedAmortizedOTQuantileRegression,
    LinearVectorQuantileRegression,
    FastNonLinearVectorQuantileRegression,
)

# yapf: disable
name_to_dataset_map: dict[str, Dataset] = {
    "banana": BananaDataset,
    "tictac": TicTacDataset,
    "convex_banana": ConvexBananaDataset,
    "not_conditional_banana": NotConditionalBananaDataset,
    "fnlvqr_mvn": FNLVQR_MVN,
}

name_to_pushforward_operator_map: dict[str, PushForwardOperator] = {
    "fast_non_linear_vector_quantile_regression": FastNonLinearVectorQuantileRegression,
    "entropic_optimal_transport_quantile_regression": EntropicOTQuantileRegression,  # Paragraph 2.4.3
    "convex_potential_flow": CPFlow,  # Paragraph 2.4.5
    "linear_vector_quantile_regression": LinearVectorQuantileRegression,
    "unconstrained_optimal_transport_quantile_regression": UnconstrainedOTQuantileRegression,  # Paragraph 2.4.4
    "unconstrained_amortized_optimal_transport_quantile_regression": UnconstrainedAmortizedOTQuantileRegression,  # Amortized varient of Paragraph 2.4.4, https://arxiv.org/abs/2210.12153
}
# yapf: enable