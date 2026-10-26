import random
import sys

import pandas as pd

from data import common
from data.common import CASE, case_attributes, encode_categorical, fit_scalers

WINDOW_SIZE = 20

# Columns rule_k: the entire trace satisfies the antecedent of rule Rk of nesy/knowledge.py (make_rule_labels.py),
# with the label implied by the rule (R1 -> 1, R2 and R3 -> 0).
RULE_TARGETS = {"rule_1": 1, "rule_2": 0, "rule_3": 0}

# Model input features, in this order.
FEATURES = ["concept:name", "case:LoanGoal", "CreditScore", "Action", "org:resource", "lifecycle:transition",
            "case:ApplicationType", "case:RequestedAmount", "FirstWithdrawalAmount", "NumberOfTerms",
            "MonthlyCost", "OfferedAmount"]
CATEGORICAL = ["concept:name", "org:resource", "Action", "lifecycle:transition", "case:LoanGoal",
               "case:ApplicationType", "NumberOfTerms"]
NUMERICAL = ["case:RequestedAmount", "FirstWithdrawalAmount", "MonthlyCost", "CreditScore", "OfferedAmount"]


import pandas as pd

from data.common import CASE, build_splits, case_attributes, encode_categorical, fit_scalers, prepare_log, split_cases

WINDOW_SIZE = 20

# Label implied by each rule when it fires (rule_1 -> 1, the others -> 0).
RULE_TARGETS = {"rule_1": 1, "rule_2": 0, "rule_3": 0, "rule_4": 0}

# Model input features, in this order. The rules in main_bpi17.py / metrics.py index them by position
# (feature i spans columns [i * WINDOW_SIZE, (i + 1) * WINDOW_SIZE)), so do not reorder.
FEATURES = ["concept:name", "case:LoanGoal", "CreditScore", "Action", "org:resource", "lifecycle:transition",
            "case:ApplicationType", "case:RequestedAmount", "FirstWithdrawalAmount", "NumberOfTerms",
            "MonthlyCost", "OfferedAmount"]
CATEGORICAL = ["concept:name", "org:resource", "Action", "lifecycle:transition", "case:LoanGoal",
               "case:ApplicationType", "NumberOfTerms"]
NUMERICAL = ["case:RequestedAmount", "FirstWithdrawalAmount", "MonthlyCost", "CreditScore", "OfferedAmount"]

# Category codes printed for reference: the rule definitions hard-code some of them.
LOGGED_CODES = {
    "concept:name": ["O_Created", "O_Create Offer", "A_Accepted", "A_Submitted", "W_Validate application",
                     "A_Complete"],
    "case:LoanGoal": ["Existing loan takeover", "Car"],
    "case:ApplicationType": ["New credit"],
}


def create_test_set(data, seed, ratio=0.2):
    """Compliance split: ~70% of the held-out cases are cases where a rule fires and agrees with the label;
    the rest are drawn from cases where no rule contradicts the label. Returns (remaining_ids, held_out_ids)."""
    rng = random.Random(seed)
    cases = case_attributes(data, ["label", *RULE_TARGETS])
    case_ids = cases.index.tolist()
    n_test = int(len(case_ids) * ratio)

    fired = cases[list(RULE_TARGETS)] == 1
    agrees = pd.DataFrame({rule: cases["label"] == target for rule, target in RULE_TARGETS.items()})
    compliant_ids = cases.index[(fired & agrees).any(axis=1)].tolist()
    non_compliant_ids = set(cases.index[(fired & ~agrees).any(axis=1)])

    print("Len training set: ", len(case_ids) - n_test)
    print("Len test set: ", n_test)
    print("Number of compliant traces: ", len(compliant_ids))
    print("Number of non-compliant traces: ", len(non_compliant_ids))

    if len(compliant_ids) > n_test:
        test_ids = rng.sample(compliant_ids, int(n_test * 0.7))
        taken = set(test_ids)
        candidates = [x for x in case_ids if x not in taken and x not in non_compliant_ids]
        test_ids += rng.sample(candidates, n_test - len(test_ids))
    else:
        test_ids = compliant_ids

    test_set = set(test_ids)
    training_ids = [x for x in case_ids if x not in test_set]

    compliant_set = set(compliant_ids)
    print("Compliant traces in training set: ", sum(x in compliant_set for x in training_ids))
    print("Compliant traces in test set: ", sum(x in compliant_set for x in test_ids))

    return training_ids, test_ids


def encode_features(data, train_ids):
    """Integer-encodes categorical features and min-max scales numerical ones (scalers fit on `train_ids`).
    Returns (events, vocab_sizes, scalers, category_codes)."""
    data = data[[CASE, "label", *FEATURES]].copy()
    data["org:resource"] = data["org:resource"].fillna("Unknown")
    data[NUMERICAL + ["NumberOfTerms"]] = data[NUMERICAL + ["NumberOfTerms"]].fillna(0)
    vocab_sizes, codes = encode_categorical(data, CATEGORICAL)
    scalers = fit_scalers(data, NUMERICAL, train_ids)
    return data, vocab_sizes, scalers, codes


def preprocess_eventlog(data, seed, setting="compliance"):
    return common.preprocess_eventlog(sys.modules[__name__], data, seed, setting)
