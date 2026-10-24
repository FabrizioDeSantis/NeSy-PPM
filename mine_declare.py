'''
Mine the Declare constraints used by main.py --declare, without training.

    python mine_declare.py --dataset sepsis
    python mine_declare.py --dataset bpi12 --setting temporal --declare_top_k 20
    python mine_declare.py --dataset traffic --declare_source predefined

The constraints are mined with pm4py on the training prefixes of the split (--setting, --seed) and selected as in
main.py (see nesy/declare_knowledge.py). They are printed with their statistics on the training prefixes and
written to results/declare_<dataset>_<setting>_<source>_seed<seed>.json.
'''
import argparse
import os

from nesy.datasets import DATASETS, load_dataset
from nesy.declare_knowledge import add_declare_args, build_declare_knowledge


def get_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--setting", default="compliance", choices=["compliance", "temporal"])
    parser.add_argument("--seed", type=int, default=42, help="Seed of the compliance split")
    parser.add_argument("--output_dir", default="results")
    add_declare_args(parser)
    return parser.parse_args()


def main():
    args = get_args()
    data = load_dataset(args.dataset, args.seed, args.setting)
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir,
                        f"declare_{args.dataset}_{args.setting}_{args.declare_source}_seed{args.seed}.json")
    build_declare_knowledge(data, args, path)


if __name__ == "__main__":
    main()
