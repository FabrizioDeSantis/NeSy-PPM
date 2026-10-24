'''
Create the rule columns rule_1, rule_2, rule_3 of the processed event logs.

    python make_rule_labels.py                              # all datasets, written to data_processed/rule_labels/
    python make_rule_labels.py --datasets sepsis bpi12      # a subset
    python make_rule_labels.py --datasets bpi12 --inplace   # overwrite data_processed/bpi12.csv (backup: .bak)

rule_k is 1 on every event of a case whose entire trace satisfies the antecedent of rule Rk of nesy/knowledge.py,
0 otherwise. The consequent (the label implied by the rule) is not considered. Only R1-R3 get a column: the additional
rules E1, ... are used for training but not by the compliance split.

The columns matter: data/preprocess_<dataset>.create_test_set builds the compliance test set from the
rule-compliant cases, so changing them changes the train / val / test split (--setting compliance) and every
result. That is why the new files are written to --output_dir by default; with --inplace the original is first
copied to <file>.bak. All the other columns (including older rule_4, rule_5, ... columns, which are no longer
used) are written back exactly as read.
'''
import argparse
import os
import shutil

import numpy as np
import pandas as pd

from data import common
from nesy.datasets import DATASETS, FeatureLayout, PreparedData, encode_traces, read_log
from nesy.knowledge import build_knowledge


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    parser.add_argument("--output_dir", default="data_processed/rule_labels")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the processed logs (after a .bak copy)")
    return parser.parse_args()


def rule_labels(name, raw):
    '''(case ids, rule names, (N, K) bool): truth of the rule antecedents on the entire trace of each case.'''
    module = DATASETS[name].preprocessing
    data = common.prepare_log(raw)
    # The rule thresholds are compared in scaled units, so the scalers can be fit on every case
    events, vocab_sizes, scalers, codes = module.encode_features(data, data[common.CASE].unique())
    layout = FeatureLayout(tuple(module.FEATURES), seq_len=1)  # seq_len is set by encode_traces
    prepared = PreparedData(name, None, None, None, layout, vocab_sizes, scalers, codes, events,
                            {"train": [], "val": [], "test": []})
    traces = encode_traces(prepared)
    prepared.layout = traces.layout
    knowledge = build_knowledge(prepared)
    rules = knowledge.split_rules
    return traces.case_ids, [r.name for r in rules], knowledge.truth(traces.x, rules).numpy()


def main():
    args = get_args()
    if not args.inplace:
        os.makedirs(args.output_dir, exist_ok=True)
    for name in args.datasets:
        path = DATASETS[name].path
        print(f"-- {name}: {path}")
        raw = read_log(name)
        text = pd.read_csv(path, dtype=str, keep_default_na=False)  # the columns as written, to keep them unchanged
        case_ids, rule_names, truth = rule_labels(name, raw)
        columns = [f"rule_{k + 1}" for k in range(len(rule_names))]
        values = pd.DataFrame(truth.astype(int), index=case_ids, columns=columns)

        cases = raw[common.CASE]
        has_case = cases.notna().to_numpy()
        for column, rule in zip(columns, rule_names):
            new = np.zeros(len(raw), dtype=int)  # events without a case id (dropped by the preprocessing): 0
            new[has_case] = values[column].reindex(cases[has_case]).fillna(0).astype(int).to_numpy()
            if column in raw:
                old_cases = raw[has_case].groupby(common.CASE)[column].first()
                new_cases = values[column].reindex(old_cases.index).fillna(0)
                changed = int((old_cases.round() != new_cases).sum())
                print(f"   {column} = {rule}: {int(new_cases.sum())} cases (before {int(old_cases.sum())}), "
                      f"{changed} of {len(old_cases)} cases changed")
            else:
                print(f"   {column} = {rule}: {int(values[column].sum())} of {len(values)} cases (new column)")
            text[column] = new

        output = path if args.inplace else os.path.join(args.output_dir, os.path.basename(path))
        if args.inplace and not os.path.exists(path + ".bak"):
            shutil.copy2(path, path + ".bak")
            print(f"   original copied to {path}.bak")
        text.to_csv(output, index=False)
        print(f"   written {output}")


if __name__ == "__main__":
    main()
