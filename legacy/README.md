# Legacy experiment scripts

These are the per-dataset `main_<dataset>.py` scripts that `main.py` replaced. They are kept for reference, along with
the code they import: `legacy/metrics.py`, `legacy/model/`, `legacy/data/dataset.py` and the original
`declare_ltn_templates.py`. `knowledge_base_old.txt` is the hand-written knowledge base they were described by.

Run them from the repository root:

```bash
PYTHONPATH=$PWD python legacy/main_sepsis.py --train_vanilla --model_type lstm
```

They use the current `data/preprocess_<dataset>.py`. The compliance split now reads only the columns `rule_1`,
`rule_2` and `rule_3`, so their results can differ from earlier runs. They still save checkpoints such as
`best_model.pth` in the working directory, overwriting the previous ones.

| Old command | New command |
|---|---|
| `python main_<dataset>.py --train_vanilla` | `python main.py --dataset <dataset> --experiments vanilla` |
| `--train_ltn_no_rules` | `--experiments ltn_no_rules` |
| `--train_ltn_no_pruning` | `--experiments ltn_no_pruning` |
| `--train_ltn_pruning` | `--experiments ltn_pruning` |
| `--train_ltn_no_pruning_weighted` | `--experiments ltn_no_pruning_weighted` |
| `--train_ltn_adaptive` | `--experiments ltn_adaptive` |
| `--model_type`, `--seed`, `--setting`, `--num_epochs`, `--num_epochs_nesy` | accepted as they are (`--seed` also takes several seeds) |

`main_traffic.py` corresponds to `--dataset traffic`. `main_traffic_old.py` is an older version of it.

## Why the results differ from `main.py`

- **Rules.** `main.py` defines the rules once, in `nesy/knowledge.py`:
  - R1-R3 of NeSyPM-Compliance: LTN axioms, compliance metric and `rule_k` columns;
  - the additional rules E1, … of these scripts: LTN axioms only. "CRP < 10" now only looks at CRP measurements.
    Before, it compared padding and missing values (stored as 0) too, so it fired on every prefix.

  The old scripts each had their own rule set with hard-coded columns and token codes, and `metrics.py` checked a
  different subset of rules. Examples:
  - the BPI12 rule "50k < amount < 60k" fired on no case;
  - the BPI17 rule `ex_loan_takeover` tested LoanGoal code 3, which is "Car";
  - the sepsis lactic acid threshold was 4 for training and in `metrics.py`, but the `rule_1` column used 2;
  - the BPI12 amount rules of `metrics.py` read columns 200-240 of a 120-column input.
- **Protocol.** Each experiment of each script had its own rules:
  - Stopping: BPI12 and Traffic stopped vanilla when the validation loss rose and tested the last weights; others
    kept the best checkpoint or trained for every epoch.
  - Learning rates differed between scripts.
  - BPI12 used a different adaptive algorithm, an EMA of rule reliability.
  - Checkpoint files were shared between scripts.
  - Sepsis and BPI17 printed the test F1 at every epoch.

  In `main.py` every experiment keeps its best validation-F1 parameters (in memory), with the tuned per-dataset
  patience, learning rates and pruning settings of `nesy/datasets.py`. Adaptive weighting uses gradient alignment
  on all datasets, and the test set is used once, at the end.
- **Determinism.** The model classes reset the global seed in their constructors. Batches were shuffled without a
  seeded generator. `main.py` seeds everything once per experiment, so every experiment of a seed starts from the
  same initialization and batch order.
- **Models.** The numerical inputs were hard-coded per dataset. They are now the features without a vocabulary,
  which changes the order of the BPI17 numerical features. The rest of the architecture is unchanged.
- **Metrics.** `metrics.py` reported accuracy, F1, precision and recall in percent for the LTN experiments.
  `main.py` reports fractions for every experiment, and also per-rule compliance and confusion matrices.
