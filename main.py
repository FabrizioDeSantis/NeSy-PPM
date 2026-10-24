'''
Train and test the experiments of the two-stage LTN with rule pruning.

    python main.py --dataset sepsis                                         # every experiment, LSTM backbone
    python main.py --dataset bpi12 --experiments vanilla ltn_pruning        # a subset
    python main.py --dataset traffic --model transformer --setting temporal --seeds 42 67 55
    python main.py --dataset bpi17 --experiments ltn_adaptive --declare in_p   # mined Declare constraints in P

Experiments (--experiments, see nesy/experiments.py): vanilla, ltn_no_rules, ltn_no_pruning, ltn_pruning,
ltn_no_pruning_weighted, ltn_adaptive. The rules of each dataset are defined once, in nesy/knowledge.py; the
tuned settings of each dataset (learning rates, patience, pruning) in nesy/datasets.py.

Declare constraints (--declare, see nesy/declare_knowledge.py): by default the 10 best constraints mined with pm4py
from the training prefixes (--declare_top_k, --declare_source predefined for the hand-written ones), included in P
as extra inputs (--declare in_p) or as axioms Forall x: C(x) (--declare axioms).

Results: results/runs/<dataset>_<model>_<timestamp>.json (arguments, rules, Declare constraints, all metrics), one
row per (seed, experiment) appended to results/results.csv, and with --results_path the metrics as JSON
({experiment: metrics} for one seed, as expected by run_seeds.py; {seed: {experiment: metrics}} otherwise).
'''
import argparse
import csv
import json
import math
import os
import platform
import statistics
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS, before CUDA is initialized

import torch

from nesy.datasets import DATASETS, load_dataset, read_log
from nesy.declare_knowledge import add_declare_args, build_declare_knowledge
from nesy.evaluation import METRICS
from nesy.experiments import EXPERIMENTS, run_experiment
from nesy.knowledge import build_knowledge


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", "--model_type", default="lstm", choices=["lstm", "transformer"],
                        help="Backbone / neural predicate P")
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS, choices=EXPERIMENTS + ["all"],
                        help="Experiments to train and test (default: all)")
    parser.add_argument("--setting", default="compliance", choices=["compliance", "temporal"],
                        help="compliance: test set built from rule-compliant cases; temporal: chronological split")
    parser.add_argument("--seeds", "--seed", type=int, nargs="+", default=[42],
                        help="One run of every experiment per seed (the compliance split also depends on it)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--results_path", default=None, help="Also write the test metrics to this JSON file")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", "--num_epochs", type=int, default=50, help="Maximum epochs of vanilla")
    training.add_argument("--epochs_ltn", "--num_epochs_nesy", type=int, default=50,
                          help="Maximum epochs of the LTN experiments")
    training.add_argument("--batch_size", type=int, default=32)

    network = parser.add_argument_group("network")
    network.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    network.add_argument("--num_layers", type=int, default=2, help="LSTM layers")
    network.add_argument("--model_dim", type=int, default=128, help="Transformer model dimension")
    network.add_argument("--transformer_layers", type=int, default=1, help="Transformer layers")
    network.add_argument("--num_heads", type=int, default=2, help="Transformer attention heads")
    network.add_argument("--dropout", "--dropout_rate", type=float, default=0.1)

    add_declare_args(parser)
    args = parser.parse_args()
    # Canonical order, without duplicates
    args.experiments = EXPERIMENTS if "all" in args.experiments else [e for e in EXPERIMENTS if e in args.experiments]
    return args


def print_table(results, experiments):
    '''Test metrics per experiment, mean ± std over the seeds.'''
    print("\n" + f"{'experiment':<25}" + "".join(f"{m:>20}" for m in METRICS))
    for name in experiments:
        row = f"{name:<25}"
        for metric in METRICS:
            values = [r[name][metric] for r in results.values() if not math.isnan(r[name][metric])]
            if not values:
                row += f"{'n/a':>20}"
            elif len(values) == 1:
                row += f"{values[0]:>20.4f}"
            else:
                row += f"{statistics.mean(values):>11.4f} ± {statistics.stdev(values):.4f}"
        print(row)


def save_results(args, knowledge, declare, results):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(args.output_dir, "runs"), exist_ok=True)
    path = os.path.join(args.output_dir, "runs", f"{args.dataset}_{args.model}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump({
            "args": vars(args),
            "versions": {"python": platform.python_version(), "torch": torch.__version__},
            "rules": [{"name": r.name, "label": r.label, "formula": r.formula, "in_split": r.split}
                      for r in knowledge.rules],
            "declare_constraints": {str(seed): [str(c) for c in d.constraints] for seed, d in declare.items()},
            "results": {str(seed): metrics for seed, metrics in results.items()},
        }, f, indent=2)

    table = os.path.join(args.output_dir, "results.csv")
    new_file = not os.path.exists(table)
    with open(table, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "dataset", "model", "setting", "declare", "seed", "experiment"] + METRICS)
        for seed, metrics in results.items():
            for name, m in metrics.items():
                writer.writerow([timestamp, args.dataset, args.model, args.setting, args.declare, seed, name]
                                + [m[k] for k in METRICS])
    print(f"\n-- Results saved to {path} and {table}")

    if args.results_path:
        payload = next(iter(results.values())) if len(results) == 1 else {str(s): m for s, m in results.items()}
        with open(args.results_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"-- Metrics saved to {args.results_path}")


def main():
    args = get_args()
    print(f"-- {args.dataset} | model {args.model} | setting {args.setting} | experiments {args.experiments} | "
          f"seeds {args.seeds} | declare {args.declare} | device {args.device}")
    raw = read_log(args.dataset)
    results, declare = {}, {}
    data = knowledge = None
    for seed in args.seeds:
        # The temporal split does not depend on the seed: the data (and the mined constraints) are shared
        if data is None or args.setting == "compliance":
            data = load_dataset(args.dataset, seed, args.setting, raw)
            knowledge = build_knowledge(data)
            knowledge.print_summary(data.train.sequences, data.train.labels)
            if args.declare != "none":
                os.makedirs(args.output_dir, exist_ok=True)
                path = os.path.join(args.output_dir, f"declare_{args.dataset}_{args.setting}_{args.declare_source}"
                                                     f"_seed{seed}.json")
                declare[seed] = build_declare_knowledge(data, args, path)
        elif args.declare != "none":
            declare[seed] = declare[args.seeds[0]]

        results[seed] = {}
        for name in args.experiments:
            print(f"\n-- Training {name} (seed {seed})")
            results[seed][name] = run_experiment(name, args, data, knowledge, declare.get(seed), seed)
            print("   test: " + ", ".join(f"{m} {results[seed][name][m]:.4f}" for m in METRICS))
    print_table(results, args.experiments)
    save_results(args, knowledge, declare, results)


if __name__ == "__main__":
    main()
