from datasets import (
    BananaDataset,
    TicTacDataset,
    Dataset,
    QuadraticPotentialConvexBananaDataset,
    FNLVQR_MVN,
    FNLVQR_Glasses,
    FNLVQR_Star,
    FNLVQR_Banana,
    NotConditionalBananaDataset,
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
    "quadratic_potential_convex_banana": QuadraticPotentialConvexBananaDataset,
    "not_conditional_banana": NotConditionalBananaDataset,
    "fnlvqr_mvn": FNLVQR_MVN,
    "fnlvqr_glasses": FNLVQR_Glasses,
    "fnlvqr_star": FNLVQR_Star,
    "fnlvqr_banana": FNLVQR_Banana,
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