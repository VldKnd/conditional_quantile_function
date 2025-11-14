from collections.abc import Callable
import functools

import os
import numpy as np
from scipy.io import arff
import pandas as pd
import pooch

from conformal.real_datasets.preprocessing import preprocess

_RAW_DATASETS = "./data/raw/"
_PROCESSED_DATASETS = "./data/processed/"

__all__ = ["loaders", "datasets"]

os.makedirs(_RAW_DATASETS, exist_ok=True)
os.makedirs(_PROCESSED_DATASETS, exist_ok=True)

# Names of all available datasets
datasets: list[str] = []
# Map from dataset name to the function that loads the dataset as a pair of NumPy arrays
loaders: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {}


def scm1d_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, :-16].values, df.iloc[:, -16:].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    return X, Y


def download_with_pooch(name: str, url: str, known_hash: str | None):
    """
    Parameterazed decorator to convert initial dataset preprocessing function
    to a loading function and polulate the registry.
    Each dataset will be downloaded and stored locally as a raw and 
    a preprocessed NumPy .npz archive with X and Y.
    """

    def _decorator(processor):

        @functools.wraps(processor)
        def wrapper() -> tuple[np.ndarray, np.ndarray]:
            file_name = pooch.retrieve(
                url=url, known_hash=known_hash, path=_RAW_DATASETS, processor=processor
            )
            with np.load(file_name) as npzf:
                X, Y = npzf["X"], npzf["Y"]
            return X, Y

        datasets.append(name)
        loaders[name] = wrapper
        return wrapper

    return _decorator


def make_pooch_precessor(file_name_processed: str):
    full_path = os.path.join(_PROCESSED_DATASETS, file_name_processed)

    def _decorator(func):

        @functools.wraps(func)
        def wrapper(fname, action, pooch):
            '''
            Processes the downloaded file and returns a new file name.

            The function **must** take as arguments (in order):

            fname : str
                The full path of the file in the local data storage
            action : str
                Either: "download" (file doesn't exist and will be downloaded),
                "update" (file is outdated and will be downloaded), or "fetch"
                (file exists and is updated so no download is necessary).
            pooch : pooch.Pooch
                The instance of the Pooch class that is calling this function.

            The return value can be anything but is usually a full path to a file
            (or list of files). This is what will be returned by Pooch.fetch and
            pooch.retrieve in place of the original file path.
            '''
            if action in ("update", "download") or not os.path.isfile(full_path):
                X, Y = func(fname)
                np.savez(full_path, X=X, Y=Y)
            return full_path

        return wrapper

    return _decorator


@download_with_pooch(
    name="rf1",
    url="https://www.openml.org/data/download/21230440/file173039e7713b.arff",
    known_hash="994f9e334811e040d2002e46d3ce06504e5d441249bf65448f53d6ea24b33cf0"
)
@make_pooch_precessor(file_name_processed="rf1.npz")
def rf1_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, :-8].values, df.iloc[:, -8:].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    # Remove outliers
    for i, j in zip(
        [4830, 4836, 4842, 4848, 4854, 4866, 4878, 4890],
        [1, 9, 17, 25, 33, 41, 49, 57]
    ):
        X[i, j] = np.median(X[:, j])
    Y[4782, 1] = np.median(Y[:, 1])

    return X, Y


@download_with_pooch(
    name="rf2",
    url="https://www.openml.org/data/download/21230441/file17307ff5552.arff",
    known_hash="b2d841515a70bbfdd212bd8bd6d138280caa4b7fc2ab16d0e3f28cb6fc2b7145"
)
@make_pooch_precessor(file_name_processed="rf2.npz")
def rf2_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, :-8].values, df.iloc[:, -8:].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    # Remove outliers
    for i, j in zip(
        [4830, 4836, 4842, 4848, 4854, 4866, 4878, 4890],
        [1, 9, 17, 25, 33, 41, 49, 57]
    ):
        X[i, j] = np.median(X[:, j])
    Y[4782, 1] = np.median(Y[:, 1])

    return X, Y


@download_with_pooch(
    name="scm1d",
    url="https://www.openml.org/data/download/21230442/file1730122322aa.arff",
    known_hash="4def7af4a1da3e20b719513d6d7e581f8b743c3d5094f7def925d53ba8a86268"
)
@make_pooch_precessor(file_name_processed="scm1d.npz")
def scm1d_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, :-16].values, df.iloc[:, -16:].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    return X, Y


@download_with_pooch(
    name="scm20d",
    url="https://www.openml.org/data/download/21230443/file1730492b4408.arff",
    known_hash="2261a94bff5cfb617dbd2e7fd2c17d15d45135103a081917d1358c2ad13a4bd2"
)
@make_pooch_precessor(file_name_processed="scm20d.npz")
def scm20d_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, :-16].values, df.iloc[:, -16:].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    return X, Y


@download_with_pooch(
    name="sgemm",
    url="https://www.openml.org/data/download/22101726/data.arff",
    known_hash="a9368120f990c92e03b47410263f28d4306d56b6cb7d2487f120a4d5e88fdb71"
)
@make_pooch_precessor(file_name_processed="sgemm.npz")
def sgemm_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    X, Y = df.iloc[:, :-4].values, df.iloc[:, -4:].values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X = imputer.fit_transform(X)

    return X, Y


@download_with_pooch(
    name="bio",
    url="https://api.openml.org/data/download/22111827/file22f167620a212.arff",
    known_hash="2b1ff847f0eafa74cd582f789273bdca9c43decf8c46910d6e2a2f51ffddb9fc"
)
@make_pooch_precessor(file_name_processed="bio.npz")
def bio_processor(fname):
    data, meta = arff.loadarff(fname)
    df = pd.DataFrame(data)
    targets = ['F7', 'F9']
    x = df[df.columns.difference(targets)]
    y = df[targets]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


@download_with_pooch(
    name="blog",
    url="https://archive.ics.uci.edu/static/public/304/blogfeedback.zip",
    known_hash="1ba74e5ad920f7cd037502b2968581cc695146a226a73eff52fe8ad875ed4bcf"
)
@make_pooch_precessor(file_name_processed="blog.npz")
def blog_processor(fname):
    fname_extracted = pooch.Unzip(members=["blogData_train.csv"]
                                  )(fname, "download", pooch=None)[0]
    df = pd.read_csv(fname_extracted, header=None)
    targets = [60, 280]
    x = df[df.columns.difference(targets)]
    y = df[targets]
    x, y, categorical_mask = preprocess(x, y)
    return x, y


@download_with_pooch(
    name="ujiindoor",
    url="https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip",
    known_hash="893512b82dfd7a7c345d84195b1c8019fbca0fa0d7820ce491ce5aa45ec3782f"
)
@make_pooch_precessor(file_name_processed="ujiindoor.npz")
def ujiindoor_processor(fname):
    fname_extracted = pooch.Unzip()(fname, "download", pooch=None)[1]
    df = pd.read_csv(fname_extracted, header=0)

    targets = ["LONGITUDE", "LATITUDE"]
    X = df[df.columns.difference(targets)]
    Y = df[targets]
    X, Y, _ = preprocess(X, Y)

    return X, Y


def read_local_dataset(
    name: str,
    csv_path: str,
    processor: Callable[[str], tuple[np.ndarray, np.ndarray]],
    processed_file_name: str | None = None,
    force: bool = False,
) -> None:
    """
    Register a local CSV-based dataset.
    - If the processed NPZ is missing (or older than the CSV, or force=True), rebuild it.
    - Adds an entry to `datasets` and `loaders` that returns (X, Y).
    """
    if processed_file_name is None:
        processed_file_name = f"{name}.npz"

    processed_path = os.path.join(_PROCESSED_DATASETS, processed_file_name)

    def _build_if_needed() -> None:
        if force or (not os.path.isfile(processed_path)) or (
            os.path.isfile(csv_path)
            and os.path.getmtime(processed_path) < os.path.getmtime(csv_path)
        ):
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(
                    f"Local dataset not found: {csv_path}. "
                    f"Place 'act_dataset.csv' next to the repo or pass a correct path."
                )
            X, Y = processor(csv_path)
            np.savez(processed_path, X=X, Y=Y)

    def _loader() -> tuple[np.ndarray, np.ndarray]:
        _build_if_needed()
        with np.load(processed_path) as npzf:
            X = npzf["X"]
            Y = npzf["Y"]
            return X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1)

    datasets.append(name)
    loaders[name] = _loader


def act_preprocesser(fname: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a locally generated ATC/ACT sequences CSV and return (X, Y).

    Expected columns (produced by your sequencing step):
      - History features: anything that's NOT in {'x_t','y_t','z_t','person_id','time_t'}
      - Targets: 'x_t', 'y_t', 'z_t'
      - Optional meta: 'person_id', 'time_t' (dropped)

    Notes:
      * Imputes missing feature values with the median (like other processors).
      * Returns float32 arrays.
    """
    df = pd.read_csv(fname)

    target_cols = ["x_t", "y_t", "z_t"]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Expected target columns {target_cols} not found. Missing: {missing}. "
            "Make sure you ran the sequencing step that creates '*_t-<k>' and 'x_t,y_t,z_t'."
        )

    # Exclude targets and common meta from features
    meta_cols = {"person_id", "time_t"}
    feature_cols = [c for c in df.columns if c not in set(target_cols) | meta_cols]

    # Build arrays
    X = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
    Y = df.loc[:, target_cols].to_numpy(dtype=np.float32, copy=False)

    # Median impute like the other processors (rf*, scm*, etc.)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    X = imputer.fit_transform(X).astype(np.float32, copy=False)

    return X, Y


read_local_dataset(
    name="act",
    csv_path=os.path.join(os.getcwd(), "act_dataset.csv"),
    processor=act_preprocesser,
    processed_file_name="act.npz",
)

read_local_dataset(
    name="act_10x10",
    csv_path=os.path.join(os.getcwd(), "act_dataset.csv"),
    processor=act_preprocesser,
    processed_file_name="act_10x10.npz",
)

if __name__ == "__main__":
    for name, loader in loaders.items():
        X, Y = loader()
        print(f"{name=}, {X.shape=}, {Y.shape=}")
        del X
        del Y
    print("Pass!")
