from __future__ import annotations

import glob
import random
from pathlib import Path
from typing import Sequence, Tuple, Iterable

import numpy as np
import pandas as pd

# ============================== #
#            CONSTANTS           #
# ============================== #

N_HISTORY = 17
N_FUTURE = 5
MAX_SAMPLES = 25_000
SAMPLE_MODE = "reservoir"  # {"reservoir", "first_k"}
SEED = 123

FEATURE_COLS: Tuple[str, ...] = ("x", "y", "z", "velocity", "angle", "facing")
TARGET_COLS: Tuple[str, ...] = ("x", "y", "z")


def make_sequence_columns(
    n_history: int,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    n_future: int = 1,  # <--- NEW
) -> tuple[list[str], list[str], list[str]]:
    """
    Build column names for flattened history features, multi-step targets, and meta columns.

    Returns:
        hist_cols: e.g., ["x_t-100", ..., "facing_t-1"]
        tgt_cols : e.g., ["x_t+1","y_t+1","z_t+1","x_t+2","y_t+2","z_t+2", ...]
        meta_cols: ["person_id", "time_t"]  (time_t = timestamp of first target step, t+1)
    """
    # History labels: t-n_history ... t-1
    hist_cols = [
        f"{feat}_t-{k}" for k in range(n_history, 0, -1) for feat in feature_cols
    ]

    # Targets for t+1 ... t+n_future (flattened in feature-major order per step)
    tgt_cols = [f"{c}_t+{k}" for k in range(1, n_future + 1) for c in target_cols]

    meta_cols = ["person_id", "time_t"]
    return hist_cols, tgt_cols, meta_cols


# ============================== #
#     PER-PERSON SEQ GENERATOR   #
# ============================== #


def iter_sequences_from_group(
    grp: pd.DataFrame,
    n_history: int,
    feature_cols: Sequence[str],
    target_cols: Sequence[str],
    n_future: int = 1,  # <--- NEW
) -> Iterable[tuple[np.ndarray, np.ndarray, int, float]]:
    """
    Yield sliding-window sequences for a single person's (already) unsorted group.

    Yields tuples:
        hist_flat (float32, shape (n_history * n_features,))
        tgt_flat  (float32, shape (n_future * n_targets,))  # <--- multi-step flattened
        pid       (int)
        time_t    (float)  # timestamp of first target step (t+1)
    """
    grp = grp.sort_values("time", kind="mergesort")

    arr_feat = grp.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)  # (n, F)
    arr_tgt = grp.loc[:, target_cols].to_numpy(dtype=np.float32, copy=False)  # (n, T)
    times = grp["time"].to_numpy(dtype=np.float64, copy=False)  # (n,)
    pid_val = int(grp["person_id"].iloc[0])

    n = len(grp)
    if n <= n_history or n_history + n_future > n:
        return

    # Last valid start so that we have history (H) + future (n_future)
    last_start = n - n_history - n_future
    for i in range(last_start + 1):
        # history covers indices [i, ..., i+H-1]; first target is at i+H (t+1)
        hist = arr_feat[i:i + n_history, :].reshape(-1)  # (H*F,)
        fut_block = arr_tgt[i + n_history:i + n_history + n_future, :]  # (n_future, T)
        tgt = fut_block.reshape(-1)  # (n_future*T,)
        tval = float(times[i + n_history])  # time of first future step (t+1)
        yield hist, tgt, pid_val, tval


# ============================== #
#     STREAMING SEQ BUILDER      #
# ============================== #


def stream_atc_to_sequences(
    path_pattern: str,
    n_history: int = N_HISTORY,
    n_future: int = N_FUTURE,  # <--- NEW
    max_samples: int = MAX_SAMPLES,
    sample_mode: str = SAMPLE_MODE,  # "first_k" or "reservoir"
    seed: int = SEED,
    feature_cols: Sequence[str] = FEATURE_COLS,
    target_cols: Sequence[str] = TARGET_COLS,
) -> pd.DataFrame:
    """
    Stream over CSV files matched by `path_pattern`, group by person, and construct
    sliding-window sequences. Build up to `max_samples` rows.

    sample_mode:
        - "first_k"   : stop once `max_samples` sequences are kept.
        - "reservoir" : maintain a uniform random subset of up to `max_samples`.

    Returns:
        DataFrame with columns:
            [history feature columns...] + [multi-step target columns...] + ["person_id", "time_t"]
    """
    if sample_mode not in {"first_k", "reservoir"}:
        raise ValueError("sample_mode must be one of {'first_k','reservoir'}.")

    if n_future < 1:
        raise ValueError("n_future must be >= 1.")

    random.seed(seed)

    files = sorted(glob.glob(path_pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {path_pattern}")

    # Source (headerless) schema
    col_names = ["time", "person_id", "x", "y", "z", "velocity", "angle", "facing"]
    dtypes = {
        "time": np.float64,
        "person_id": np.int64,
        "x": np.int32,
        "y": np.int32,
        "z": np.float32,
        "velocity": np.float32,
        "angle": np.float32,
        "facing": np.float32,
    }

    # Prepare reservoir buffers
    hist_cols, tgt_cols, meta_cols = make_sequence_columns(
        n_history, feature_cols, target_cols, n_future=n_future
    )

    n_feat_vals = n_history * len(feature_cols)
    n_tgt_vals = n_future * len(target_cols)  # <--- multi-step
    total_feat = n_feat_vals + n_tgt_vals

    X = np.empty((max_samples, total_feat), dtype=np.float32)
    P = np.empty(max_samples, dtype=np.int64)  # person_id
    T = np.empty(max_samples, dtype=np.float64)  # time_t (first target time)

    seen = 0  # total sequences observed
    kept = 0  # sequences stored (<= max_samples)

    def maybe_store(row_feat: np.ndarray, pid: int, tval: float) -> bool:
        """
        Returns:
            True  -> continue streaming
            False -> stop early (only used by first_k)
        """
        nonlocal seen, kept
        if sample_mode == "first_k":
            if kept < max_samples:
                X[kept, :] = row_feat
                P[kept] = pid
                T[kept] = tval
                kept += 1
                return True
            return False  # stop early

        # reservoir sampling
        if kept < max_samples:
            X[kept, :] = row_feat
            P[kept] = pid
            T[kept] = tval
            kept += 1
        else:
            j = random.randint(0, seen)
            if j < max_samples:
                X[j, :] = row_feat
                P[j] = pid
                T[j] = tval
        return True

    for f in files:
        df = pd.read_csv(f, header=None, names=col_names, dtype=dtypes, engine="c")

        # Deduplicate within-file by (person_id, time)
        if not df.empty:
            df = df.drop_duplicates(subset=["person_id", "time"], keep="last")

        for _, grp in df.groupby("person_id", sort=False):
            for hist, tgt, pid_val, tval in iter_sequences_from_group(
                grp, n_history, feature_cols, target_cols, n_future=n_future
            ):
                # concat -> float32 row vector
                row_feat = np.concatenate([hist, tgt]).astype(np.float32, copy=False)

                cont = maybe_store(row_feat, pid_val, tval)
                seen += 1
                if not cont:
                    cols = hist_cols + tgt_cols
                    out = pd.DataFrame(X[:kept, :], columns=cols)
                    out["person_id"] = P[:kept]
                    out["time_t"] = T[:kept]
                    return out

    # Finished streaming
    cols = hist_cols + tgt_cols
    out = pd.DataFrame(X[:kept, :], columns=cols)
    out["person_id"] = P[:kept]
    out["time_t"] = T[:kept]
    return out


# ============================== #
#     TEMPORAL TRAIN/TEST SPLIT  #
# ============================== #


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_col: str = "time_t",
    gap_seconds: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split ensuring max(train[time_col]) <= min(test[time_col]) (with optional gap).

    Notes:
    - test_size is approximate if many rows share the boundary timestamp.
    - If `gap_seconds` > 0, we enforce: min(test[time_col]) >= max(train[time_col]) + gap_seconds
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found.")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1).")

    n = len(df)
    if n == 0:
        return df.copy(), df.copy()

    s = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    cut_idx = int(np.floor((1.0 - test_size) * n))
    cut_idx = min(max(cut_idx, 1), n - 1)

    cutoff_time = s.loc[cut_idx - 1, time_col]
    train = s[s[time_col] < cutoff_time]

    # Fallback when all boundary rows share the same timestamp
    if train.empty:
        train = s.iloc[:cut_idx]
        test = s.iloc[cut_idx:]
        return train.reset_index(drop=True), test.reset_index(drop=True)

    if gap_seconds > 0:
        boundary = float(train[time_col].max()) + float(gap_seconds)
        test = s[s[time_col] >= boundary]
    else:
        test = s[s[time_col] >= cutoff_time]

    return train.reset_index(drop=True), test.reset_index(drop=True)


# ============================== #
#          SAVE TO NPZ           #
# ============================== #


def save_dataset_npz(
    df: pd.DataFrame,
    file_name: str = "atc_one_day_100x5.npz",  # <--- updated default name
    n_history: int = N_HISTORY,
    n_future: int = N_FUTURE,  # <--- NEW
    feature_cols: Sequence[str] = FEATURE_COLS,
    target_cols: Sequence[str] = TARGET_COLS,
    reshape_3d:
    bool = True,  # True -> X: (N, n_history, len(feature_cols)); Y: (N, n_future, len(target_cols))
    dtype: str = "float32",
) -> str:
    """
    Save dataset (built by the sequencing step) to a .npz with keys X (history) and Y (multi-step targets).

    df: must include history columns like 'x_t-100'...'facing_t-1' and targets
        'x_t+1','y_t+1','z_t+1', ... up to n_future.
    """
    # Normalize filename and ensure parent directory exists
    if not file_name.endswith(".npz"):
        file_name = file_name + ".npz"
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)

    hist_cols, tgt_cols, _ = make_sequence_columns(
        n_history, feature_cols, target_cols, n_future=n_future
    )

    # Validate presence
    missing = [c for c in (hist_cols + tgt_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns (first few): {missing[:10]}")

    X_flat = df.loc[:, hist_cols].to_numpy(dtype=dtype, copy=False)
    Y_flat = df.loc[:, tgt_cols].to_numpy(dtype=dtype, copy=False)

    if reshape_3d:
        X = X_flat.reshape((-1, n_history, len(feature_cols)))
        Y = Y_flat.reshape((-1, n_future, len(target_cols)))
    else:
        X = X_flat  # (N, n_history * len(feature_cols))
        Y = Y_flat  # (N, n_future * len(target_cols))

    np.savez_compressed(file_name, X=X, Y=Y)
    return file_name
