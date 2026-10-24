import random
import sys

from data import common
from data.common import CASE, case_attributes, encode_categorical, fit_scalers, sample_test_cases

WINDOW_SIZE = 40

# Columns rule_k: the entire trace satisfies the antecedent of rule Rk of nesy/knowledge.py (make_rule_labels.py).
# R1-R3 all imply rejection (label 0).
RULES = ["rule_1", "rule_2", "rule_3"]

# Model input features, in this order.
FEATURES = ["org:resource", "concept:name", "case:AMOUNT_REQ"]
CATEGORICAL = ["concept:name", "org:resource"]
NUMERICAL = ["case:AMOUNT_REQ"]


def create_test_set(data, seed, ratio=0.2):
    """Compliance split: held-out cases are drawn from those where the rules agree with the label
    (a rule fires and the case is rejected, or no rule fires and the case is accepted).
    Returns (remaining_ids, held_out_ids)."""
    rng = random.Random(seed)
    cases = case_attributes(data, ["label", *RULES])
    fired = (cases[RULES] == 1).any(axis=1)
    none_fired = (cases[RULES] == 0).all(axis=1)
    compliant = fired & (cases["label"] == 0)
    pool = cases.index[compliant | (none_fired & (cases["label"] == 1))]
    print("Len filtered values: ", len(pool))

    training_ids, test_ids = sample_test_cases(cases.index.tolist(), pool, int(len(cases) * ratio), rng)

    compliant_ids = set(cases.index[compliant])
    print("Len training set: ", len(training_ids))
    print("Len test set: ", len(test_ids))
    print("Number of compliant traces in training set: ", sum(x in compliant_ids for x in training_ids))
    print("Number of compliant traces in test set: ", sum(x in compliant_ids for x in test_ids))
    return training_ids, test_ids


def encode_features(data, train_ids):
    """Integer-encodes categorical features and min-max scales numerical ones (scalers fit on `train_ids`).
    Returns (events, vocab_sizes, scalers, category_codes)."""
    data = data[[CASE, "label", *FEATURES]].copy()
    data[NUMERICAL] = data.groupby(CASE)[NUMERICAL].ffill()
    vocab_sizes, codes = encode_categorical(data, CATEGORICAL)
    scalers = fit_scalers(data, NUMERICAL, train_ids)
    return data, vocab_sizes, scalers, codes


def preprocess_eventlog(data, seed, setting="compliance"):
    return common.preprocess_eventlog(sys.modules[__name__], data, seed, setting)
