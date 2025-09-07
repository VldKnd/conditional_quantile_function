from datasets import (
    BananaDataset, TicTacDataset, Dataset, QuadraticPotentialConvexBananaDataset,
    FNLVQR_MVN, FNLVQR_Glasses, FNLVQR_Star, FNLVQR_Banana, NotConditionalBananaDataset,
    PICNN_FNLVQR_Banana, PICNN_FNLVQR_Glasses, PICNN_FNLVQR_Star, FunnelDistribution
)
from pushforward_operators import (
    PushForwardOperator,
    CPFlow,
    EntropicNeuralQuantileRegression,
    AmortizedNeuralQuantileRegression,
    NeuralQuantileRegression,
    LinearQuantileRegression,
    FastNonLinearVectorQuantileRegression,
    MeanQuantileRegression,
)

# yapf: disable
name_to_dataset_map: dict[str, Dataset] = {
    "banana": BananaDataset,
    "tictac": TicTacDataset,
    "funnel":FunnelDistribution,
    "quadratic_potential_convex_banana": QuadraticPotentialConvexBananaDataset,
    "not_conditional_banana": NotConditionalBananaDataset,
    "fnlvqr_mvn": FNLVQR_MVN,
    "fnlvqr_glasses": FNLVQR_Glasses,
    "fnlvqr_star": FNLVQR_Star,
    "fnlvqr_banana": FNLVQR_Banana,
    "picnn_fnlvqr_banana": PICNN_FNLVQR_Banana,
    "picnn_fnlvqr_glasses": PICNN_FNLVQR_Glasses,
    "picnn_fnlvqr_star": PICNN_FNLVQR_Star,
}

name_to_pushforward_operator_map: dict[str, PushForwardOperator] = {
    "fast_non_linear_vector_quantile_regression": FastNonLinearVectorQuantileRegression,
    "entropic_neural_quantile_regression": EntropicNeuralQuantileRegression,
    "convex_potential_flow": CPFlow,
    "linear_quantile_regression": LinearQuantileRegression,
    "neural_quantile_regression": NeuralQuantileRegression,
    "amortized_neural_quantile_regression": AmortizedNeuralQuantileRegression,
    "mean_quantile_regression": MeanQuantileRegression,
}
# yapf: enable
