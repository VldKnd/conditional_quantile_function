from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from conformal.real_datasets.process_raw import loaders


@dataclass
class SplitParameters:
    dataset_name: str
    idx_train: np.ndarray
    idx_cal: np.ndarray
    idx_test: np.ndarray


@dataclass
class DatasetSplit:
    X_train: np.ndarray = field(init=False)
    Y_train: np.ndarray = field(init=False)
    X_cal: np.ndarray = field(init=False)
    Y_cal: np.ndarray = field(init=False)
    X_test: np.ndarray = field(init=False)
    Y_test: np.ndarray = field(init=False)
    X_train_pre_pca: np.ndarray = field(init=False)
    X_cal_pre_pca: np.ndarray = field(init=False)
    X_test_pre_pca: np.ndarray = field(init=False)
    X_train_raw: np.ndarray
    Y_train_raw: np.ndarray
    X_cal_raw: np.ndarray
    Y_cal_raw: np.ndarray
    X_test_raw: np.ndarray
    Y_test_raw: np.ndarray
    n_features: int = field(init=False)
    n_outputs: int = field(init=False)
    n_train: int = field(init=False)
    n_cal: int = field(init=False)
    n_test: int = field(init=False)
    X_scaler: StandardScaler = field(init=False)
    Y_scaler: StandardScaler = field(init=False)
    reduce: bool = False
    pca: PCA = field(init=False)

    def __post_init__(self):
        self.n_train, self.n_features = self.X_train_raw.shape
        self.n_cal = self.X_cal_raw.shape[0]
        self.n_test, self.n_outputs = self.Y_test_raw.shape

        self.X_scaler = StandardScaler().fit(self.X_train_raw)
        self.Y_scaler = StandardScaler().fit(self.Y_train_raw)

        self.X_train, self.X_cal, self.X_test = \
            map(self.X_scaler.transform, (self.X_train_raw,
                                          self.X_cal_raw,
                                          self.X_test_raw))
        self.Y_train, self.Y_cal, self.Y_test = \
            map(self.Y_scaler.transform, (self.Y_train_raw,
                                          self.Y_cal_raw,
                                          self.Y_test_raw))

        if self.reduce and self.n_features > 50:
            n_components = 50 if self.n_features < 150 else 100
            self.pca = PCA(n_components=n_components)
            self.pca.fit(self.X_train)
            self.X_train_pre_pca, self.X_cal_pre_pca, self.X_test_pre_pca = \
                self.X_train, self.X_cal, self.X_test
            self.X_train, self.X_cal, self.X_test = \
            map(self.pca.transform, (self.X_train,
                                     self.X_cal,
                                     self.X_test))
            self.n_features = n_components


def get_dataset_split(
    name: str,
    seed: int,
    n_train=None,
    n_cal=2000,
    n_test=2000,
    reduce=True
) -> DatasetSplit:
    load_func = loaders.get(name, None)
    if load_func is not None:
        X, Y = load_func()
        n_total = X.shape[0]

        # TODO: add logic to set only some of the sizes?
        if n_train is None:
            assert n_cal is not None and n_test is not None
            n_train = n_total - n_cal - n_test

        train_start, train_end = 0, n_train
        cal_start, cal_end = n_train, n_train + n_cal
        test_start, test_end = n_train + n_cal, n_train + n_cal + n_test

        if seed >= 0:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(n_total)
        else:
            idx = np.arange(n_total)
        idx_train = idx[train_start:train_end]
        idx_cal = idx[cal_start:cal_end]
        idx_test = idx[test_start:test_end]

        return DatasetSplit(
            X_train_raw=X[idx_train],
            Y_train_raw=Y[idx_train],
            X_cal_raw=X[idx_cal],
            Y_cal_raw=Y[idx_cal],
            X_test_raw=X[idx_test],
            Y_test_raw=Y[idx_test],
            reduce=reduce,
        )
    else:
        raise Exception(f"Unknown dataset: {name}.")


if __name__ == "__main__":
    ds = get_dataset_split("rf1", seed=42)
    print(ds)
