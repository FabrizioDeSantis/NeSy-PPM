import random
import sys

from data import common
from data.common import CASE, case_attributes, encode_categorical, fit_scalers, sample_test_cases

WINDOW_SIZE = 10

# Columns rule_k: the entire trace satisfies the antecedent of rule Rk of nesy/knowledge.py (make_rule_labels.py).
# R1-R3 all imply credit collection (label 1).
RULES = ["rule_1", "rule_2", "rule_3"]

# Model input features, in this order.
FEATURES = ["concept:name", "org:resource", "dismissal", "vehicleClass", "article", "points", "expense",
            "notificationType", "lastSent", "amount", "paymentAmount"]
CATEGORICAL = ["concept:name", "org:resource", "dismissal", "lastSent", "points", "vehicleClass",
               "notificationType", "article"]
NUMERICAL = ["amount", "expense"]

# Carried over from earlier events of the same case when missing.
FORWARD_FILLED = ["dismissal", "amount", "article"]
FILL_VALUES = {"org:resource": 0, "vehicleClass": 0, "notificationType": 0, "lastSent": 0, "points": -1,
               "amount": 0, "paymentAmount": 0, "expense": 0}


from data.common import (CASE, build_splits, case_attributes, encode_categorical, fit_scalers, prepare_log,
                         sample_test_cases, split_cases)

WINDOW_SIZE = 10

# Rules used to build the compliance split; each one implies label 1.
RULES = ["rule_1", "rule_2", "rule_3", "rule_4"]

# Model input features, in this order. The rules in main_traffic.py / metrics.py index them by position
# (feature i spans columns [i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE)), so do not reorder.
FEATURES = ["concept:name", "org:resource", "dismissal", "vehicleClass", "article", "points", "expense",
            "notificationType", "lastSent", "amount", "paymentAmount"]
CATEGORICAL = ["concept:name", "org:resource", "dismissal", "lastSent", "points", "vehicleClass",
               "notificationType", "article"]
NUMERICAL = ["amount", "expense"]

# Carried over from earlier events of the same case when missing.
FORWARD_FILLED = ["dismissal", "amount", "article"]
FILL_VALUES = {"org:resource": 0, "vehicleClass": 0, "notificationType": 0, "lastSent": 0, "points": -1,
               "amount": 0, "paymentAmount": 0, "expense": 0}

# Category codes printed for reference: the rule definitions hard-code some of them.
LOGGED_CODES = {
    "concept:name": ["Add penalty", "Create Fine", "Send Fine", "Payment", "Insert Fine Notification"],
    "vehicleClass": ["A", "C", "M", "R"],
}


def create_test_set(data, seed, ratio=0.2):
    """Compliance split: held-out cases are drawn from half of the cases where each rule fires with label 1
    (the other half stays in training) and from the cases where no rule fires and the label is 0.
    Returns (remaining_ids, held_out_ids)."""
    rng = random.Random(seed)
    cases = case_attributes(data, ["label", *RULES])
    pool = set(cases.index[(cases[RULES] == 0).all(axis=1) & (cases["label"] == 0)])
    compliant_ids = set()
    for rule in RULES:
        rule_ids = cases.index[(cases[rule] == 1) & (cases["label"] == 1)].tolist()
        kept_for_training = set(rng.sample(rule_ids, len(rule_ids) // 2))
        pool.update(x for x in rule_ids if x not in kept_for_training)
        compliant_ids.update(rule_ids)

    training_ids, test_ids = sample_test_cases(cases.index.tolist(), pool, int(len(cases) * ratio), rng)

    print("Number of traces in training set: ", len(training_ids))
    print("Number of traces in test set: ", len(test_ids))
    print("Number of compliant traces in training set: ", sum(x in compliant_ids for x in training_ids))
    print("Number of compliant traces in test set: ", sum(x in compliant_ids for x in test_ids))
    return training_ids, test_ids


def encode_features(data, train_ids):
    """Integer-encodes categorical features and min-max scales numerical ones (scalers fit on `train_ids`).
    Payments are scaled with the fine-amount scaler, so the two stay comparable.
    Returns (events, vocab_sizes, scalers, category_codes)."""
    data = data[[CASE, "label", *FEATURES]].copy()
    data[FORWARD_FILLED] = data.groupby(CASE)[FORWARD_FILLED].ffill()
    data = data.fillna(FILL_VALUES)
    vocab_sizes, codes = encode_categorical(data, CATEGORICAL)
    scalers = fit_scalers(data, NUMERICAL, train_ids)
    scalers["paymentAmount"] = scalers["amount"]
    data["paymentAmount"] = scalers["amount"].transform(data[["paymentAmount"]].to_numpy(dtype=float))[:, 0]
    return data, vocab_sizes, scalers, codes


def preprocess_eventlog(data, seed, setting="compliance"):
    return common.preprocess_eventlog(sys.modules[__name__], data, seed, setting)
