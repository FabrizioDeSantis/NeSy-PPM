'''Datasets of the experiments: where they are, their tuned training settings, and how they are fed to the models.'''
from collections import Counter
from dataclasses import dataclass, field
from types import ModuleType
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import common, preprocess_bpi12, preprocess_bpi17, preprocess_sepsis, preprocess_traffic
from data.dataset import NeSyDataset


@dataclass(frozen=True)
class Training:
    '''Optimizer and early stopping of one experiment.'''
    lr: float = 1e-3
    patience: Optional[int] = None  # epochs without validation-F1 improvement; None: no early stopping


@dataclass(frozen=True)
class Pruning:
    '''Rule pruning of ltn_pruning (see nesy/experiments.RulePruner).'''
    stats_epoch: int         # epoch during which the rule statistics are collected
    gate_epoch: int          # rules are scored at the end of this epoch; afterwards only reliable rules stay active
    threshold: float         # a rule stays active if its reliability score is > threshold
    variance_penalty: float
    fired_only: bool         # score a rule only on the prefixes where its antecedent fires
    no_support_score: float  # score of a rule firing on fewer than 2 prefixes (fired_only)


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    preprocessing: ModuleType  # data/preprocess_<dataset>.py: split and encoding
    pruning: Pruning
    training: dict = field(default_factory=dict)  # experiment -> Training (default: Training())


# Tuned values of the former main_<dataset>.py scripts (legacy/).
DATASETS = {
    "bpi12": DatasetSpec(
        "data_processed/bpi12.csv", preprocess_bpi12,
        Pruning(stats_epoch=5, gate_epoch=5, threshold=0.25, variance_penalty=1.0, fired_only=True,
                no_support_score=1.0),
        {"vanilla": Training(patience=5), "ltn_pruning": Training(patience=8),
         "ltn_no_pruning_weighted": Training(patience=15), "ltn_adaptive": Training(patience=15)}),
    "bpi17": DatasetSpec(
        "data_processed/bpi17.csv", preprocess_bpi17,
        Pruning(stats_epoch=0, gate_epoch=0, threshold=0.4, variance_penalty=2.0, fired_only=True,
                no_support_score=0.0),
        {"vanilla": Training(patience=5), "ltn_pruning": Training(patience=5),
         "ltn_no_pruning_weighted": Training(patience=5), "ltn_adaptive": Training(patience=15)}),
    "sepsis": DatasetSpec(
        "data_processed/sepsis_2.csv", preprocess_sepsis,
        Pruning(stats_epoch=3, gate_epoch=3, threshold=0.3, variance_penalty=1.0, fired_only=False,
                no_support_score=0.0),
        {"vanilla": Training(patience=8), "ltn_no_pruning": Training(lr=1e-4),
         "ltn_pruning": Training(lr=1e-4, patience=8), "ltn_no_pruning_weighted": Training(patience=8),
         "ltn_adaptive": Training(lr=1e-4, patience=10)}),
    "traffic": DatasetSpec(
        "data_processed/traffic_fines.csv", preprocess_traffic,
        Pruning(stats_epoch=5, gate_epoch=5, threshold=0.6, variance_penalty=2.0, fired_only=False,
                no_support_score=0.0),
        {"vanilla": Training(patience=5), "ltn_no_pruning": Training(patience=5),
         "ltn_pruning": Training(patience=5), "ltn_no_pruning_weighted": Training(patience=15),
         "ltn_adaptive": Training(patience=15)}),
}


@dataclass(frozen=True)
class FeatureLayout:
    '''Position of the features in the flat model input: feature i occupies columns [i * L, (i + 1) * L).'''
    feature_names: tuple
    seq_len: int

    @property
    def width(self):
        return len(self.feature_names) * self.seq_len

    def get(self, x, name):
        '''(B, L) values of feature `name` at every event of the prefixes in x (0 on padding).'''
        start = self.feature_names.index(name) * self.seq_len
        return x[:, start:start + self.seq_len]


@dataclass
class PreparedData:
    name: str
    train: NeSyDataset
    val: NeSyDataset
    test: NeSyDataset
    layout: FeatureLayout
    vocab_sizes: dict     # categorical feature -> number of codes (codes start at 1, 0 is padding)
    scalers: dict         # numerical feature -> MinMaxScaler fitted on the training cases
    category_codes: dict  # categorical feature -> {value: code}
    events: pd.DataFrame  # encoded event log, sorted by case and time
    split_ids: dict       # "train" / "val" / "test" -> case ids


@dataclass
class TraceSet:
    '''Entire traces (all the events of each case), encoded like the prefixes, padded to the longest trace.'''
    x: torch.Tensor  # (N, F * L)
    y: torch.Tensor  # (N,)
    split: np.ndarray  # (N,) "train", "val", "test" or None
    layout: FeatureLayout
    case_ids: np.ndarray  # (N,)


def read_log(name):
    return pd.read_csv(DATASETS[name].path, dtype={"org:resource": str})


def load_dataset(name, seed, setting, raw=None):
    '''Read, split and encode a dataset, and build the prefixes of each split.'''
    spec = DATASETS[name]
    if raw is None:
        print(f"-- Reading {spec.path}")
        raw = read_log(name)
    log = common.preprocess(spec.preprocessing, raw, seed, setting)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = common.build_splits(
        log.events, log.split_ids, log.features, log.window_size)
    data = PreparedData(name, NeSyDataset(X_train, y_train), NeSyDataset(X_val, y_val), NeSyDataset(X_test, y_test),
                        FeatureLayout(tuple(feature_names), log.window_size), log.vocab_sizes, log.scalers,
                        log.category_codes, log.events, dict(zip(("train", "val", "test"), log.split_ids)))
    print_summary(data)
    return data


def encode_traces(data):
    '''The entire trace of every case, in the feature-major layout of the prefixes.'''
    events = data.events.reset_index(drop=True)
    case_index, case_ids = pd.factorize(events[common.CASE], sort=True)
    position = events.groupby(case_index).cumcount().to_numpy()
    names = list(data.layout.feature_names)
    x = np.zeros((len(case_ids), len(names), position.max() + 1), dtype=np.float32)
    x[case_index, :, position] = events[names].to_numpy(dtype=np.float32)
    y = events.groupby(case_index)["label"].first().to_numpy(dtype=np.float32)
    split_of = {case: split for split, ids in data.split_ids.items() for case in ids}
    split = np.array([split_of.get(case) for case in case_ids], dtype=object)
    layout = FeatureLayout(data.layout.feature_names, seq_len=x.shape[2])
    return TraceSet(torch.from_numpy(x).flatten(1), torch.from_numpy(y), split, layout, np.asarray(case_ids))


def print_summary(data):
    print(f"-- Dataset {data.name}: {len(data.layout.feature_names)} features x {data.layout.seq_len} events")
    print(f"   features: {list(data.layout.feature_names)}")
    for split in ("train", "val", "test"):
        dataset = getattr(data, split)
        counts = Counter(int(label) for label in dataset.labels.tolist())
        print(f"   {split}: {len(data.split_ids[split])} cases, {len(dataset)} prefixes, "
              f"labels {dict(sorted(counts.items()))}")


def make_loader(dataset, batch_size, shuffle=False, seed=None):
    '''With shuffle, the batch order is fixed by seed.'''
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)
