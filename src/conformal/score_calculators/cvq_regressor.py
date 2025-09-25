from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, Type
import warnings

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.ensemble import RandomForestRegressor
import torch
from torch.func import hessian
from torch.utils.data import TensorDataset, DataLoader

from conformal.real_datasets.reproducible_split import DatasetSplit
from conformal.score_calculators.protocol import ScoreCalculator
from conformal.score_calculators.selected_params import selected_params
from infrastructure.classes import TrainParameters
from pushforward_operators.convex_potential_flow.core_flow import ConvexPotentialFlow
from pushforward_operators.protocol import PushForwardOperator
from pushforward_operators import NeuralQuantileRegression, AmortizedNeuralQuantileRegression


def _make_xy_dataloader(
    X: np.ndarray, Y: np.ndarray, batch_size: int, dtype=torch.float64
) -> DataLoader:
    dataset = TensorDataset(torch.tensor(X, dtype=dtype), torch.tensor(Y, dtype=dtype))
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


@dataclass
class BaseVQRegressor(ScoreCalculator):

    feature_dimension: int
    response_dimension: int
    hidden_dimension: int
    number_of_hidden_layers: int
    batch_size: int
    n_epochs: int
    learning_rate: float = 0.01
    potential_to_estimate_with_neural_network = "u"
    dtype: torch.dtype = torch.float64
    betas: tuple[float, float] = (0.5, 0.5)
    weight_decay: float = 1e-4
    warmup_iterations: int = 5
    model: PushForwardOperator = field(init=False)

    def __post_init__(self):
        print(f"{self.potential_to_estimate_with_neural_network=}")
        _optimizer_parameters = dict(
            lr=self.learning_rate, betas=self.betas, weight_decay=self.weight_decay
        )
        self.train_parameters = TrainParameters(
            number_of_epochs_to_train=self.n_epochs,
            optimizer_parameters=_optimizer_parameters,
            scheduler_parameters={"eta_min": 0.},
            verbose=True,
            warmup_iterations=self.warmup_iterations,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        dataloader = _make_xy_dataloader(
            X, Y, batch_size=self.batch_size, dtype=self.dtype
        )
        self.model.train()
        self.model.fit(
            dataloader,
            train_parameters=self.train_parameters,
        )
        self.model.eval()

    def predict_mean(self, X: np.ndarray):
        n = X.shape[0]
        U = torch.zeros((n, self.response_dimension), dtype=self.dtype)
        X_tensor = torch.tensor(X, dtype=self.dtype)
        self.model.eval()
        Y = self.model.push_u_given_x(x=X_tensor, u=U)
        return Y.numpy(force=True)

    def predict_quantile(self, X: np.ndarray, Y: np.ndarray):
        u = self.model.push_y_given_x(
            y=torch.tensor(Y, dtype=self.dtype), x=torch.tensor(X, dtype=self.dtype)
        ).numpy(force=True)
        self.model.eval()
        return u

    def predict_inverse_quantile(self, X: np.ndarray, U: np.ndarray):
        y = self.model.push_u_given_x(
            u=torch.tensor(U, dtype=self.dtype), x=torch.tensor(X, dtype=self.dtype)
        ).numpy(force=True)
        self.model.eval()
        return y

    @classmethod
    def _train_or_load(
        cls,
        pf_cls: Type[PushForwardOperator],
        save_path: Path,
        model_config,
        X_train: np.ndarray,
        Y_train: np.ndarray,
    ) -> Self:
        print(f"{save_path=}")
        n_features = X_train.shape[1]
        n_outputs = Y_train.shape[1]

        reg_cvqr = cls(
            feature_dimension=n_features, response_dimension=n_outputs, **model_config
        )
        # Fit base models
        if Path.is_file(save_path):
            #reg_cvqr.model.load(trained_model_path_cvqr)
            reg_cvqr.model = pf_cls.load_class(str(save_path))
        else:
            reg_cvqr.fit(X_train, Y_train)
            reg_cvqr.model.save(str(save_path))
        return reg_cvqr


@dataclass
class CVQRegressor(BaseVQRegressor):
    ckpt_name_old = "model_cvqr.pth"

    def __post_init__(self):
        super().__post_init__()
        self.model = AmortizedNeuralQuantileRegression(
            feature_dimension=self.feature_dimension,
            response_dimension=self.response_dimension,
            hidden_dimension=self.hidden_dimension,
            number_of_hidden_layers=self.number_of_hidden_layers,
            potential_to_estimate_with_neural_network=self.
            potential_to_estimate_with_neural_network,
        ).to(self.dtype)

    def compute_logdet_hessian(
        self,
        condition: torch.Tensor,
        tensor: torch.Tensor,
        batch_size: int | None = None
    ):
        _compute_batch_hessian = torch.vmap(
            func=hessian(self.model.potential_network, argnums=1),
            in_dims=0,
            chunk_size=batch_size or self.batch_size
        )
        hessians = _compute_batch_hessian(condition, tensor)[:, 0].detach()
        logdet_hessians = torch.logdet(hessians).squeeze().detach().numpy(force=True)
        return logdet_hessians

    def calculate_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Calculate Monge-Kantorovich qunatiles and ranks for given sample pairs (x_i, y_i).
        Additionaly, calculate estimate of the log-density fro the given samples
        Returns: quantiles, ranks, log_p

        """
        _, d = Y.shape
        self.model.eval()
        quantiles = self.predict_quantile(X, Y)
        ranks = np.linalg.norm(quantiles, axis=-1)
        X_tensor = torch.tensor(X, dtype=self.dtype)
        Y_tensor = torch.tensor(Y, dtype=self.dtype)

        if self.potential_to_estimate_with_neural_network == "y":
            logdet_hessians = self.compute_logdet_hessian(
                condition=X_tensor, tensor=Y_tensor, batch_size=batch_size
            )

            log_density_in_u = multivariate_normal.logpdf(quantiles, mean=np.zeros(d))
            log_density = log_density_in_u + logdet_hessians

        if self.potential_to_estimate_with_neural_network == "u":
            logdet_hessians = self.compute_logdet_hessian(
                condition=X_tensor,
                tensor=torch.tensor(quantiles).to(X_tensor),
                batch_size=batch_size
            )

            log_density_in_u = multivariate_normal.logpdf(quantiles, mean=np.zeros(d))
            log_density = log_density_in_u - logdet_hessians

        return {"MK Quantile": quantiles, "MK Rank": ranks, "Log Density": log_density}

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        ckpt_path = path / f"model_{str(cls.__name__)}.pth"
        if not ckpt_path.is_file():
            ckpt_path = path / cls.ckpt_name_old
        return cls._train_or_load(
            pf_cls=AmortizedNeuralQuantileRegression,
            save_path=ckpt_path,
            model_config=selected_params[args.dataset],
            X_train=dataset_split.X_train,
            Y_train=dataset_split.Y_train
        )


@dataclass
class CVQRegressorY(CVQRegressor):
    potential_to_estimate_with_neural_network = "y"
    ckpt_name_old = "model_cvqr_y.pth"


@dataclass
class CPFlowRegressor(BaseVQRegressor):
    n_blocks: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.model = ConvexPotentialFlow(
            feature_dimension=self.feature_dimension,
            response_dimension=self.response_dimension,
            hidden_dimension=self.hidden_dimension,
            number_of_hidden_layers=self.number_of_hidden_layers,
            n_blocks=self.n_blocks,
        ).to(self.dtype)

    def calculate_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Calculate Monge-Kantorovich qunatiles and ranks for given sample pairs (x_i, y_i).
        Additionaly, calculate estimate of the log-density fro the given samples
        Returns: quantiles, ranks, log_p

        """
        n, m = X.shape
        _, d = Y.shape

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.eval()
            quantiles = self.predict_quantile(X, Y)
            ranks = np.linalg.norm(quantiles, axis=-1)
            self.model.eval()
            log_p = self.model.logp_cond(
                Y=torch.tensor(Y, dtype=self.dtype),
                X=torch.tensor(X, dtype=self.dtype)
            ).numpy(force=True)
            self.model.eval()
        return {"MK Quantile": quantiles, "MK Rank": ranks, "Log Density": log_p}

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        return cls._train_or_load(
            pf_cls=ConvexPotentialFlow,
            save_path=path / f"model_{str(cls.__name__)}.pth",
            model_config=selected_params[args.dataset],
            X_train=dataset_split.X_train,
            Y_train=dataset_split.Y_train
        )


@dataclass
class CVQRegressorRF(ScoreCalculator):
    cvqr: CVQRegressor
    rf: RandomForestRegressor

    def calculate_scores(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        batch_size: int | None = None
    ) -> dict[str, np.ndarray]:
        return self.cvqr.calculate_scores(
            X, Y - self.rf.predict(X), batch_size=batch_size
        )

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        # Number of points to reserve for training the random forest base moidel
        n = int(0.25 * dataset_split.n_train)
        rf = RandomForestRegressor(random_state=args.seed, n_jobs=args.n_cpus)
        rf.fit(dataset_split.X_train[:n], dataset_split.Y_train[:n])

        cvqr = CVQRegressor._train_or_load(
            pf_cls=AmortizedNeuralQuantileRegression,
            save_path=path / f"model_{cls.__name__}.pth",
            model_config=selected_params[args.dataset],
            X_train=dataset_split.X_train[n:],
            Y_train=dataset_split.Y_train[n:] - rf.predict(dataset_split.X_train[n:])
        )

        return cls(cvqr, rf)


@dataclass
class CVQRegressorYRF(CVQRegressorRF):
    cvqr: CVQRegressorY

    @classmethod
    def create_or_load(cls, path: Path, args, dataset_split: DatasetSplit) -> Self:
        # Number of points to reserve for training the random forest base moidel
        n = int(0.25 * dataset_split.n_train)
        rf = RandomForestRegressor(random_state=args.seed, n_jobs=args.n_cpus)
        rf.fit(dataset_split.X_train[:n], dataset_split.Y_train[:n])

        cvqr = CVQRegressorY._train_or_load(
            pf_cls=AmortizedNeuralQuantileRegression,
            save_path=path / f"model_{cls.__name__}.pth",
            model_config=selected_params[args.dataset],
            X_train=dataset_split.X_train[n:],
            Y_train=dataset_split.Y_train[n:] - rf.predict(dataset_split.X_train[n:])
        )

        return cls(cvqr, rf)


if __name__ == "__main__":
    cvqr = CVQRegressor(
        feature_dimension=1,
        response_dimension=1,
        hidden_dimension=1,
        number_of_hidden_layers=1,
        batch_size=1,
        n_epochs=1
    )
    print(cvqr.potential_to_estimate_with_neural_network)
    cvqr_y = CVQRegressorY(
        feature_dimension=1,
        response_dimension=1,
        hidden_dimension=1,
        number_of_hidden_layers=1,
        batch_size=1,
        n_epochs=1
    )
    print(cvqr_y.potential_to_estimate_with_neural_network)
