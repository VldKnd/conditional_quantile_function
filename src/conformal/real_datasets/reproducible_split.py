from dataclasses import dataclass, field

import numpy as np

from conformal.real_datasets.process_raw import loaders


@dataclass
class SplitParameters:
    dataset_name: str
    idx_train: np.ndarray
    idx_cal: np.ndarray
    idx_test: np.ndarray


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    Y_train: np.ndarray
    X_cal: np.ndarray
    Y_cal: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    n_features: int = field(init=False)
    n_train: int = field(init=False)
    n_cal: int = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self):
        self.n_train, self.n_features = self.X_train.shape
        self.n_cal = self.X_cal.shape[0]
        self.n_test = self.X_test.shape[0]


def get_dataset_split(name: str, seed:int, n_train=2000, n_cal=2000, n_test=2000):
    load_func = loaders.get(name, None)
    if load_func is not None:
        X, Y = load_func()
        n_total = X.shape[0]

        # TODO: add logic to set only some of the sizes?
        train_start, train_end = 0, n_train
        cal_start, cal_end = n_train, n_train + n_cal
        test_start, test_end = n_train + n_cal, n_train + n_cal + n_test

        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_total)
        idx_train = idx[train_start:train_end]
        idx_cal = idx[cal_start:cal_end]
        idx_test = idx[test_start:test_end]
        
        return DatasetSplit(X_train=X[idx_train], Y_train=Y[idx_train],
                            X_cal=X[idx_cal], Y_cal=Y[idx_cal],
                            X_test=X[idx_test], Y_test=Y[idx_test])
    else:
        raise Exception(f"Unknown dataset: {name}.")


if __name__ == "__main__":
    ds = get_dataset_split("rf1", seed=42)
    print(ds)
