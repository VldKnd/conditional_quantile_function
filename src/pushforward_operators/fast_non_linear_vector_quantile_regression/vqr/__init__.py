# isort: skip_file
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr.cvqf import VQFBase, DiscreteVQFBase, VQF, CVQF, DiscreteVQF, DiscreteCVQF
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr.solvers import VQRSolver
from pushforward_operators.fast_non_linear_vector_quantile_regression.vqr.api import VectorQuantileEstimator, VectorQuantileRegressor
from pkg_resources import DistributionNotFound, get_distribution

try:
    dist_name = "vqr"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
