"""Helpers shared by the dataset-specific preprocessing scripts (preprocess_<dataset>.py)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

CASE = "case:concept:name"
TIMESTAMP = "time:timestamp"


def prepare_log(data):
    """Parses timestamps and orders events by case, then time (ties keep their order in the file)."""
    data = data.dropna(subset=[CASE])
    data = data.assign(**{TIMESTAMP: pd.to_datetime(data[TIMESTAMP], format="ISO8601", utc=True)})
    print("Number of traces: ", data[CASE].nunique())
    return data.sort_values([CASE, TIMESTAMP], kind="stable")


def case_attributes(data, columns):
    """One row per case (indexed and sorted by case id) holding the first value of each of `columns`."""
    return data.groupby(CASE)[list(columns)].first()


def sample_test_cases(case_ids, pool, n_test, rng):
    """Draws `n_test` test cases from `pool` (the whole pool if it is not larger). Returns (training_ids, test_ids),
    where training_ids are the remaining `case_ids`."""
    pool = sorted(pool)
    test_ids = rng.sample(pool, n_test) if len(pool) > n_test else pool
    test_set = set(test_ids)
    return [x for x in case_ids if x not in test_set], test_ids


def create_temporal_split(data, test_ratio=0.2, val_ratio=0.2, drop_overlapping=True):
    """Chronological case split: the oldest cases go to training, the following ones to validation and the most
    recent ones to test (cases are ordered by their first event; `data[TIMESTAMP]` must be datetime).

    With `drop_overlapping`, training (validation) cases that are still running when the validation (test) period
    starts are discarded, so no event used for training or model selection happens after the start of the period
    it is evaluated on."""
    span = data.groupby(CASE)[TIMESTAMP].agg(["min", "max"]).sort_values("min", kind="stable")
    n_cases = len(span)
    n_test = int(n_cases * test_ratio)
    n_val = int((n_cases - n_test) * val_ratio)
    n_train = n_cases - n_test - n_val

    train = span.iloc[:n_train]
    val = span.iloc[n_train:n_train + n_val]
    test = span.iloc[n_train + n_val:]
    print(f"Total traces: {n_cases}, Train: {n_train}, Val: {n_val}, Test: {n_test}")

    if drop_overlapping:
        val_start, test_start = val["min"].min(), test["min"].min()
        kept_train = train[train["max"] < val_start]
        kept_val = val[val["max"] < test_start]
        print(f"Dropped {len(train) - len(kept_train)} train traces overlapping the validation period and "
              f"{len(val) - len(kept_val)} validation traces overlapping the test period")
        train, val = kept_train, kept_val

    for name, part in (("Train", train), ("Val", val), ("Test", test)):
        print(f"{name} period: {part['min'].min()} -> {part['max'].max()}")

    return train.index.tolist(), val.index.tolist(), test.index.tolist()


def split_cases(data, seed, setting, create_test_set):
    """Returns (train_ids, val_ids, test_ids). `create_test_set(data, seed, ratio)` is the dataset's compliance
    split, applied once for test and once more on the remaining cases for validation."""
    if setting == "compliance":
        train_ids, test_ids = create_test_set(data, seed)
        train_ids, val_ids = create_test_set(data[data[CASE].isin(train_ids)], seed, ratio=0.2)
    elif setting == "temporal":
        train_ids, val_ids, test_ids = create_temporal_split(data)
    else:
        raise ValueError(f"Unknown setting {setting!r}: expected 'compliance' or 'temporal'")

    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in val set: ", len(val_ids))
    print("Number of traces in test set: ", len(test_ids))
    return train_ids, val_ids, test_ids


def encode_categorical(data, columns):
    """Integer-encodes `columns` in place (codes start at 1; 0 is shared by padding and missing values).
    Returns the vocabulary sizes and, per column, the code of each value ({column: {value: code}})."""
    vocab_sizes, codes = {}, {}
    for col in columns:
        categorical = pd.Categorical(data[col])
        codes[col] = {value: i + 1 for i, value in enumerate(categorical.categories)}
        data[col] = categorical.codes + 1
        vocab_sizes[col] = data[col].max()
    return vocab_sizes, codes


def fit_scalers(data, columns, train_ids):
    """Min-max scales `columns` in place, fitting each scaler on the training cases only. Returns the scalers."""
    train_rows = data[CASE].isin(train_ids).to_numpy()
    scalers = {}
    for col in columns:
        values = data[col].to_numpy(dtype=float).reshape(-1, 1)
        scalers[col] = MinMaxScaler().fit(values[train_rows])
        data[col] = scalers[col].transform(values)[:, 0]
    return scalers


def create_ngrams(data, case_ids, features, window_size):
    """All prefixes of length 1 .. min(len, window_size) - 1 of each case, as feature-major lists
    (one list of `window_size` values per feature, zero-padded on the right), with the case label."""
    prefixes, labels = [], []
    for _, case in data[data[CASE].isin(case_ids)].groupby(CASE):
        events = case[features].to_numpy(dtype=float)[:window_size]
        label = int(case["label"].iloc[0])
        for n in range(1, len(events)):
            prefix = np.zeros((len(features), window_size))
            prefix[:, :n] = events[:n].T
            prefixes.append(prefix.tolist())
            labels.append(label)
    return prefixes, labels


def build_splits(data, split_ids, features, window_size):
    """(X_train, y_train, X_val, y_val, X_test, y_test, feature_names) for `split_ids` = (train, val, test)."""
    splits = []
    for ids in split_ids:
        splits.extend(create_ngrams(data, ids, features, window_size))
    return (*splits, list(features))


@dataclass
class ProcessedLog:
    """Encoded event log of one dataset, before the prefixes are built."""
    events: pd.DataFrame   # one row per event, sorted by case and time, features encoded
    split_ids: tuple       # (train, val, test) case ids
    vocab_sizes: dict      # categorical feature -> number of codes
    scalers: dict          # numerical feature -> MinMaxScaler fitted on the training cases
    category_codes: dict   # categorical feature -> {value: code}
    features: list         # model input features, in order
    window_size: int       # maximum prefix length


def preprocess(module, data, seed, setting):
    """Split and encode `data` with the dataset's preprocessing module (data/preprocess_<dataset>.py)."""
    data = prepare_log(data)
    split_ids = split_cases(data, seed, setting, module.create_test_set)
    events, vocab_sizes, scalers, codes = module.encode_features(data, split_ids[0])
    return ProcessedLog(events, split_ids, vocab_sizes, scalers, codes, list(module.FEATURES), module.WINDOW_SIZE)


def preprocess_eventlog(module, data, seed, setting):
    """((X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers)."""
    log = preprocess(module, data, seed, setting)
    return build_splits(log.events, log.split_ids, log.features, log.window_size), log.vocab_sizes, log.scalers
