import random
import sys

from data import common
from data.common import CASE, case_attributes, encode_categorical, fit_scalers, sample_test_cases

WINDOW_SIZE = 13

# Columns rule_k: the entire trace satisfies the antecedent of rule Rk of nesy/knowledge.py (make_rule_labels.py).
# R1-R3 all imply ICU admission (label 1).
RULES = ["rule_1", "rule_2", "rule_3"]

# Model input features, in this order.
FEATURES = ["InfectionSuspected", "DiagnosticBlood", "DisfuncOrg", "SIRSCritTachypnea", "Hypotensie",
            "SIRSCritHeartRate", "Infusion", "DiagnosticArtAstrup", "concept:name", "Age", "DiagnosticIC",
            "DiagnosticSputum", "DiagnosticLiquor", "DiagnosticOther", "SIRSCriteria2OrMore", "DiagnosticXthorax",
            "SIRSCritTemperature", "DiagnosticUrinaryCulture", "SIRSCritLeucos", "Oligurie", "DiagnosticLacticAcid",
            "Diagnose", "Hypoxie", "DiagnosticUrinarySediment", "DiagnosticECG", "Leucocytes", "CRP", "LacticAcid"]
CATEGORICAL = ["concept:name", "Diagnose"]
NUMERICAL = ["LacticAcid", "CRP", "Leucocytes", "Age"]
# Boolean flags (and Age) are stored as integers, missing values as 0.
INTEGER = [f for f in FEATURES if f not in CATEGORICAL + NUMERICAL] + ["Age"]


def create_test_set(data, seed, ratio=0.2):
    """Compliance split: held-out cases are drawn from those where the rules agree with the label
    (a rule fires and the label is 1, or no rule fires and the label is 0). Returns (remaining_ids, held_out_ids)."""
    rng = random.Random(seed)
    cases = case_attributes(data, ["label", *RULES])
    fired = (cases[RULES] == 1).any(axis=1)
    compliant = fired & (cases["label"] == 1)
    pool = cases.index[compliant | ((cases[RULES] == 0).all(axis=1) & (cases["label"] == 0))]
    n_test = int(len(cases) * ratio)

    print("Len training set: ", len(cases) - n_test)
    print("Len test set: ", n_test)
    print("Number of compliant traces: ", len(pool))
    print("Number of non-compliant traces: ", (fired & (cases["label"] == 0)).sum())

    training_ids, test_ids = sample_test_cases(cases.index.tolist(), pool, n_test, rng)

    compliant_ids = set(cases.index[compliant])
    print("Number of compliant traces in training set: ", sum(x in compliant_ids for x in training_ids))
    print("Number of compliant traces in test set: ", sum(x in compliant_ids for x in test_ids))
    return training_ids, test_ids


def drop_after_release(data):
    """For label-0 cases, keeps only the events before the first Release activity (data sorted by case and time)."""
    released = (data["label"] == 0) & data["concept:name"].str.contains(r"\bRelease\b", na=False)
    return data[~released.groupby(data[CASE]).cummax()]


def encode_features(data, train_ids):
    """Integer-encodes categorical features and min-max scales numerical ones (scalers fit on `train_ids`).
    Returns (events, vocab_sizes, scalers, category_codes)."""
    data = drop_after_release(data)[[CASE, "label", *FEATURES]].copy()
    data[INTEGER] = data[INTEGER].fillna(0).astype(int)
    data[NUMERICAL] = data[NUMERICAL].fillna(0)
    vocab_sizes, codes = encode_categorical(data, CATEGORICAL)
    scalers = fit_scalers(data, NUMERICAL, train_ids)
    return data, vocab_sizes, scalers, codes


def preprocess_eventlog(data, seed, setting="compliance"):
    return common.preprocess_eventlog(sys.modules[__name__], data, seed, setting)
