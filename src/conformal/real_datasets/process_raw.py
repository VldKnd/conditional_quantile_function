import os
import numpy as np
from scipy.io import arff
import pandas as pd
import pooch

#MULTIDIM_DATASETS = pooch.create(
#    path="./data/raw/", # pooch.os_cache("conditional_quantile_function"),
#    base_url="",
#)

_RAW_DATASETS = "./data/raw/"
_PROCESSED_DATASETS = "./data/processed/"

__all__ = ["loaders", "datasets"]


def load_rf1() -> tuple[np.ndarray, np.ndarray]:
    file_name = pooch.retrieve(
        url="https://www.openml.org/data/download/21230440/file173039e7713b.arff",
        known_hash="994f9e334811e040d2002e46d3ce06504e5d441249bf65448f53d6ea24b33cf0",
        path=_RAW_DATASETS,
        processor=rf1_processor
    )
    with np.load(file_name) as npzf:
        X, Y = npzf["X"], npzf["Y"]
    return X, Y


def rf1_processor(fname, action, pooch):
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
    full_path = os.path.join(_PROCESSED_DATASETS, "rf1.npz")
    if action in ("update", "download"):
        df = arff.loadarff(fname)
        df = pd.DataFrame(df[0])
        X, Y = df.iloc[:, :-8].values, df.iloc[:, -8:].values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(X)
        np.savez(full_path, X=X, Y=Y)

    return full_path


loaders = {
    "rf1": load_rf1,
}


datasets = tuple(loaders.keys())


if __name__ == "__main__":
    X, Y = load_rf1()
    print(f"{X.shape=}, {Y.shape=}")
