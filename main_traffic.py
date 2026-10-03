import argparse
import math
import statistics
import warnings
from collections import Counter, namedtuple
from dataclasses import dataclass

import ltn
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from data import preprocess_traffic
from data.dataset import ModelConfig, NeSyDataset
from metrics import compute_metrics
from model.lstm import LSTMModel
from model.transformer import EventTransformer

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATASET = "traffic_fines"           # name passed to compute_metrics and used for the data file
CONFIG_DATASET_NAME = "traffic"     # name stored in ModelConfig
DATA_PATH = f"data_processed/{DATASET}.csv"
BATCH_SIZE = 32
SEQUENCE_LENGTH = 10
CHECKPOINT_PATH = "best_model.pth"

# Loss mixing: loss = 1 - (DATA_WEIGHT * data_sat + KNOWLEDGE_WEIGHT * knowledge_sat)
DATA_WEIGHT = 0.8
KNOWLEDGE_WEIGHT = 0.2

# The vanilla baseline uses its own fixed learning rate; the LTN variants use config.learning_rate.
VANILLA_LEARNING_RATE = 1e-3

# Early-stopping patience (epochs without validation-F1 improvement); no_rules never early-stops
NO_PRUNING_PATIENCE = 5
PRUNING_PATIENCE = 5
WEIGHTED_PATIENCE = 10
ADAPTIVE_PATIENCE = 15

# Rule pruning
PRUNING_CALIBRATION_EPOCH = 5   # rule statistics are collected in this epoch; pruning starts after it
PRUNING_GATE_THRESHOLD = 0.6    # a rule stays active if its gate is >= this value
PRUNING_VARIANCE_PENALTY = 2.0

# Adaptive rule weighting
ADAPTIVE_EMA_DECAY = 0.8
ADAPTIVE_UNIFORM_MIX = 0.1      # share of the weights that stays uniform
ADAPTIVE_VARIANCE_PENALTY = 2.0


def block(index):
    """Columns of the `index`-th feature: each feature spans SEQUENCE_LENGTH consecutive columns."""
    return slice(index * SEQUENCE_LENGTH, (index + 1) * SEQUENCE_LENGTH)


# Feature blocks used by the rules (indices in the preprocessed layout; see `feature_names`).
ACTIVITIES = block(0)
VEHICLE_CLASS = block(3)
AMOUNT = block(9)          # fine amount per event (scaled with scalers["amount"])
PAID_AMOUNT = block(10)    # compared against the fine amount at the first event

TOKEN_ADD_PENALTY = 1
TOKEN_PAYMENT = 7
TOKEN_VEHICLE_CLASS_A = 2
HIGH_AMOUNT = 400

# --------------------------------------------------------------------------- #
# Fuzzy-logic operators (shared by all LTN experiments)
# --------------------------------------------------------------------------- #

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
SatAgg = ltn.fuzzy_ops.SatAgg()


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the LSTM model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the LSTM model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the LSTM model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Number of epochs for training LTN model")
    parser.add_argument("--model_type", type=str, default="transformer", help="Type of model: lstm or transformer")
    parser.add_argument("--train_vanilla", action="store_true", help="Train vanilla model")
    parser.add_argument("--train_ltn_no_rules", action="store_true", help="Train LTN model without rules")
    parser.add_argument("--train_ltn_no_pruning", action="store_true", help="Train LTN model without pruning")
    parser.add_argument("--train_ltn_pruning", action="store_true", help="Train LTN model with rule pruning")
    parser.add_argument("--train_ltn_no_pruning_weighted", action="store_true",
                        help="Train LTN model without pruning, with weighted loss")
    parser.add_argument("--train_ltn_adaptive", action="store_true",
                        help="Train LTN model with adaptive rule weighting")
    parser.add_argument("--setting", type=str, default="compliance",
                        help="Setting for the experiment (compliance or temporal)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


Rule = namedtuple("Rule", ["name", "antecedent"])


def indicator(fn):
    """Wrap a boolean-valued function of the batch as an ltn.Function returning 0./1. floats."""
    return ltn.Function(func=lambda x: fn(x).float())


def build_rules(scalers):
    """Background-knowledge rules `antecedent(x) -> P(x)`."""
    high_amount = scalers["amount"].transform([[HIGH_AMOUNT]])[0][0]

    add_penalty = indicator(lambda x: (x[:, ACTIVITIES] == TOKEN_ADD_PENALTY).any(dim=1))
    payment_less_than_fine = indicator(
        lambda x: (x[:, ACTIVITIES] == TOKEN_PAYMENT).any(dim=1)
        & (x[:, PAID_AMOUNT].max(dim=1).values < x[:, AMOUNT.start]))   # AMOUNT.start: fine at first event
    amount_above_400 = indicator(lambda x: (x[:, AMOUNT] > high_amount).any(dim=1))
    vehicle_class_a = indicator(lambda x: (x[:, VEHICLE_CLASS] == TOKEN_VEHICLE_CLASS_A).any(dim=1))

    return [
        Rule("add_penalty", add_penalty),
        Rule("payment_less_than_fine", payment_less_than_fine),
        Rule("amount_above_400", amount_above_400),
        Rule("vehicle_class_a", vehicle_class_a),
    ]


@dataclass
class Context:
    """Everything an experiment needs: data, config, device and rules."""
    args: argparse.Namespace
    config: ModelConfig
    device: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    vocab_sizes: dict
    feature_names: list
    scalers: dict
    rules: list

    def build_model(self):
        if self.args.model_type == "transformer":
            model = EventTransformer(
                self.vocab_sizes, self.config, self.feature_names,
                model_dim=128, num_classes=1, max_len=self.config.sequence_length,
                num_layers=1, num_heads=2, dropout=0.1,
            )
        else:
            model = LSTMModel(self.vocab_sizes, self.config, 1, self.feature_names)
        return model.to(self.device)

    def validation_f1(self, model):
        with torch.no_grad():
            model.eval()
            _, f1, _, _, _ = compute_metrics(
                self.val_loader, model, self.device, "nesy", self.scalers, DATASET)
        return f1

    def test_metrics(self, model, mode):
        model.eval()
        accuracy, f1, precision, recall, compliance = compute_metrics(
            self.test_loader, model, self.device, mode, self.scalers, DATASET)
        return {"Accuracy": accuracy, "F1 Score": f1, "Precision": precision,
                "Recall": recall, "Compliance": compliance}


def build_context(args):
    config = ModelConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        num_epochs=args.num_epochs,
        dataset=CONFIG_DATASET_NAME,
        sequence_length=SEQUENCE_LENGTH,
        seed=args.seed,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("-- Reading dataset")
    event_log = pd.read_csv(DATA_PATH, dtype={"org:resource": str})
    splits, vocab_sizes, scalers = preprocess_traffic.preprocess_eventlog(event_log, args.seed, args.setting)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = splits

    print("--- Label distribution")
    print("--- Training set")
    print(Counter(y_train))
    print("--- Test set")
    print(Counter(y_test))
    print(feature_names)

    return Context(
        args=args,
        config=config,
        device=device,
        train_loader=DataLoader(NeSyDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True),
        val_loader=DataLoader(NeSyDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False),
        test_loader=DataLoader(NeSyDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False),
        vocab_sizes=vocab_sizes,
        feature_names=feature_names,
        scalers=scalers,
        rules=build_rules(scalers),
    )


def report(title, metrics):
    print(f"Metrics {title}")
    for name, value in metrics.items():
        print(f"{name}:", value)


class BestCheckpoint:
    """Saves the weights with the best validation F1 and tracks early-stopping patience."""

    def __init__(self, model, patience, path):
        self.model = model
        self.patience = patience
        self.path = path
        self.best_f1 = -1.0
        self.epochs_without_improvement = 0

    def update(self, val_f1):
        """Record this epoch's validation F1. Returns True if training should stop."""
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.epochs_without_improvement = 0
            torch.save(self.model.state_dict(), self.path)
        else:
            self.epochs_without_improvement += 1
        return self.patience is not None and self.epochs_without_improvement >= self.patience

    def restore(self):
        self.model.load_state_dict(torch.load(self.path))


# --------------------------------------------------------------------------- #
# Vanilla (non-LTN) baseline
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, criterion, optimizer, device):
    losses = []
    for x, y in loader:
        optimizer.zero_grad()
        output = model(x.to(device))
        loss = criterion(output.squeeze(1).cpu(), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return statistics.mean(losses)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    """Returns (mean loss, true labels, predicted labels) over `loader`."""
    model.eval()
    losses, labels, predictions = [], [], []
    for x, y in loader:
        output = model(x.to(device))
        losses.append(criterion(output.squeeze(1).cpu(), y).item())
        predictions.append((output.cpu().numpy() > 0.5).astype(float).flatten())
        labels.append(y.numpy())
    return statistics.mean(losses), np.concatenate(labels), np.concatenate(predictions)


def run_vanilla(ctx):
    model = ctx.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=VANILLA_LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    val_losses = []
    for epoch in range(ctx.config.num_epochs):
        model.train()
        train_loss = train_one_epoch(model, ctx.train_loader, criterion, optimizer, ctx.device)
        print(f"Epoch {epoch + 1}/{ctx.config.num_epochs}, Loss: {train_loss}")

        val_loss, _, _ = evaluate_loader(model, ctx.val_loader, criterion, ctx.device)
        print(f"Validation Loss: {val_loss}")
        val_losses.append(val_loss)

        if epoch >= 5 and val_losses[-1] > val_losses[-2]:
            print("Validation loss increased, stopping training")
            break

    _, y_true, y_pred = evaluate_loader(model, ctx.test_loader, criterion, ctx.device)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
    }
    report("vanilla", metrics)
    return metrics


# --------------------------------------------------------------------------- #
# Shared LTN building blocks
# --------------------------------------------------------------------------- #

def supervised_formulas(P, x, y):
    """Data axioms: positives satisfy P, negatives satisfy not P."""
    formulas = []
    positives, negatives = x[y == 1], x[y == 0]
    if positives.numel() > 0:
        x_pos = ltn.Variable("x_P", positives)
        formulas.append(Forall(x_pos, P(x_pos)))
    if negatives.numel() > 0:
        x_neg = ltn.Variable("x_not_P", negatives)
        formulas.append(Forall(x_neg, Not(P(x_neg))))
    return formulas


def rule_formulas(rules, P, x_all):
    """Knowledge axioms: for every rule, antecedent(x) -> P(x)."""
    return [Forall(x_all, Implies(rule.antecedent(x_all), P(x_all))) for rule in rules]


def mixed_loss(data_sat, knowledge_sat):
    return 1 - (DATA_WEIGHT * data_sat + KNOWLEDGE_WEIGHT * knowledge_sat)


def rule_reliability(antecedent, consequent, variance_penalty, fired_only):
    """Score in [0, 1] of how well `antecedent -> consequent` holds on the recorded samples.

    score = mean(implication) * exp(-variance_penalty * var(implication)).
    With `fired_only`, the statistics use only samples where the antecedent is satisfied;
    otherwise all samples count (so rarely-firing rules score close to 1).
    """
    implication = Implies(antecedent, consequent).value
    if fired_only:
        fired = antecedent.value > 0.5
        if fired.sum() < 2:  # not enough evidence to estimate mean and variance
            return 0.0
        implication = implication[fired]
    score = implication.mean().item() * math.exp(-variance_penalty * implication.var().item())
    return max(0.0, min(1.0, score))


def weighted_p_mean_error(satisfactions, weights, p=2):
    """Weighted p-mean-error aggregator: 1 - (sum_i w_i * (1 - s_i)^p)^(1/p)."""
    weighted_errors = weights * (1 - satisfactions) ** p
    return 1 - weighted_errors.sum() ** (1 / p)


def train_ltn(ctx, model, batch_loss, *, patience=None, checkpoint_path=CHECKPOINT_PATH,
              on_epoch_end=None):
    """Generic LTN training loop.

    batch_loss(P, epoch, x, y) -> scalar loss tensor for one batch.
    Every epoch is validated; the best-F1 weights are kept and restored at the end.
    Early stopping is applied only if `patience` is given.
    on_epoch_end(epoch, model) -> optional hook run after validation.
    """
    P = ltn.Predicate(model).to(ctx.device)
    optimizer = torch.optim.Adam(P.parameters(), lr=ctx.config.learning_rate)
    checkpoint = BestCheckpoint(model, patience, checkpoint_path)

    for epoch in range(ctx.args.num_epochs_nesy):
        model.train()
        epoch_loss = 0.0
        for x, y in ctx.train_loader:
            x = x.to(ctx.device)
            optimizer.zero_grad()
            loss = batch_loss(P, epoch, x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f" epoch {epoch} | loss {epoch_loss / len(ctx.train_loader):.4f}")

        val_f1 = ctx.validation_f1(model)
        print("Validation F1 Score:", val_f1)
        if checkpoint.update(val_f1):
            print("Early stopping triggered")
            break
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)

    checkpoint.restore()


def run_ltn(ctx, title, mode, batch_loss, **train_kwargs):
    model = ctx.build_model()
    train_ltn(ctx, model, batch_loss, **train_kwargs)
    metrics = ctx.test_metrics(model, mode)
    report(title, metrics)
    return metrics


# --------------------------------------------------------------------------- #
# LTN experiments
# --------------------------------------------------------------------------- #

def run_ltn_no_rules(ctx):
    def batch_loss(P, epoch, x, y):
        return 1 - SatAgg(*supervised_formulas(P, x, y))

    return run_ltn(ctx, "LTN w/o rules", "ltn", batch_loss)


def run_ltn_no_pruning(ctx):
    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        formulas = supervised_formulas(P, x, y) + rule_formulas(ctx.rules, P, x_all)
        return 1 - SatAgg(*formulas)

    return run_ltn(ctx, "LTN w/o pruning", "ltn_w_k", batch_loss,
                   patience=NO_PRUNING_PATIENCE)


def run_ltn_no_pruning_weighted(ctx):
    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        knowledge_sat = SatAgg(*rule_formulas(ctx.rules, P, x_all))
        return mixed_loss(data_sat, knowledge_sat)

    return run_ltn(ctx, "LTN w/o pruning weighted", "ltn_w_k", batch_loss,
                   patience=WEIGHTED_PATIENCE)


class RulePruner:
    """Drops rules that look unreliable after a calibration epoch.

    * Epochs 0..calibration_epoch: every rule is active.
    * During `calibration_epoch`: record each rule's antecedent truth values and P(x).
    * At the end of `calibration_epoch`: score each rule; afterwards only rules whose
      score is >= `threshold` stay active.
    """

    def __init__(self, rules, calibration_epoch, threshold):
        self.rules = rules
        self.calibration_epoch = calibration_epoch
        self.threshold = threshold
        self.gates = [0.0] * len(rules)
        self._antecedent_truth = [[] for _ in rules]
        self._p_truth = []

    def active_rules(self, epoch):
        if epoch <= self.calibration_epoch:
            return list(self.rules)
        return [rule for rule, gate in zip(self.rules, self.gates) if gate >= self.threshold]

    def observe(self, P, epoch, x_all):
        if epoch != self.calibration_epoch:
            return
        with torch.no_grad():
            for store, rule in zip(self._antecedent_truth, self.rules):
                store.append(rule.antecedent(x_all).value)
            self._p_truth.append(P(x_all).value)

    def update_gates(self, epoch):
        if epoch != self.calibration_epoch:
            return
        with torch.no_grad():
            consequent = ltn.LTNObject(torch.cat(self._p_truth), ["x_All"])
            print(f"Gating scores after epoch {epoch}:")
            for i, (rule, truth) in enumerate(zip(self.rules, self._antecedent_truth)):
                antecedent = ltn.LTNObject(torch.cat(truth), ["x_All"])
                self.gates[i] = rule_reliability(
                    antecedent, consequent, PRUNING_VARIANCE_PENALTY, fired_only=False)
                print(f"  {rule.name}: {self.gates[i]}")


def run_ltn_pruning(ctx):
    pruner = RulePruner(ctx.rules, PRUNING_CALIBRATION_EPOCH, PRUNING_GATE_THRESHOLD)

    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        pruner.observe(P, epoch, x_all)
        active = pruner.active_rules(epoch)
        if not active:
            return 1 - data_sat
        return mixed_loss(data_sat, SatAgg(*rule_formulas(active, P, x_all)))

    return run_ltn(ctx, "LTN w pruning", "ltn_w_k", batch_loss,
                   patience=PRUNING_PATIENCE,
                   on_epoch_end=lambda epoch, model: pruner.update_gates(epoch))


class AdaptiveRuleWeights:
    """Per-rule weights re-estimated every epoch from how well each rule holds.

    Raw reliability scores are smoothed with an exponential moving average, normalised,
    and mixed with a uniform floor so no rule is ever fully switched off.
    """

    def __init__(self, rules, device):
        self.rules = rules
        self.device = device
        self.weights = torch.full((len(rules),), 1 / len(rules), device=device)
        self._ema = None
        self._reset()

    def _reset(self):
        self._antecedent_truth = [[] for _ in self.rules]
        self._p_truth = []

    def observe(self, P, x_all):
        with torch.no_grad():
            for store, rule in zip(self._antecedent_truth, self.rules):
                store.append(rule.antecedent(x_all).value)
            self._p_truth.append(P(x_all).value)

    def update(self):
        with torch.no_grad():
            consequent = ltn.LTNObject(torch.cat(self._p_truth), ["x_All"])
            raw_scores = torch.tensor(
                [rule_reliability(ltn.LTNObject(torch.cat(truth), ["x_All"]), consequent,
                                  ADAPTIVE_VARIANCE_PENALTY, fired_only=True)
                 for truth in self._antecedent_truth],
                device=self.device)

            if self._ema is None:
                self._ema = raw_scores.clone()
            else:
                self._ema = ADAPTIVE_EMA_DECAY * self._ema + (1 - ADAPTIVE_EMA_DECAY) * raw_scores

            n_rules = len(self.rules)
            if self._ema.sum().item() > 0:
                self.weights = (ADAPTIVE_UNIFORM_MIX / n_rules
                                + (1 - ADAPTIVE_UNIFORM_MIX) * self._ema / self._ema.sum())
            else:
                self.weights = torch.full_like(self._ema, 1 / n_rules)

            print("Rule weights:", self.weights.cpu().tolist())
            self._reset()


def run_ltn_adaptive(ctx):
    adaptive = AdaptiveRuleWeights(ctx.rules, ctx.device)

    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        rule_sats = torch.stack([f.value for f in rule_formulas(ctx.rules, P, x_all)])
        knowledge_sat = weighted_p_mean_error(rule_sats, adaptive.weights)
        adaptive.observe(P, x_all)
        return mixed_loss(data_sat, knowledge_sat)

    return run_ltn(ctx, "LTN w adaptive rule weights", "ltn_w_k", batch_loss,
                   patience=ADAPTIVE_PATIENCE,
                   on_epoch_end=lambda epoch, model: adaptive.update())


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    args = get_args()
    ctx = build_context(args)

    experiments = [
        (args.train_vanilla, "vanilla", run_vanilla),
        (args.train_ltn_no_rules, "ltn_no_rules", run_ltn_no_rules),
        (args.train_ltn_no_pruning, "ltn_no_pruning", run_ltn_no_pruning),
        (args.train_ltn_pruning, "ltn_pruning", run_ltn_pruning),
        (args.train_ltn_no_pruning_weighted, "ltn_no_pruning_weighted", run_ltn_no_pruning_weighted),
        (args.train_ltn_adaptive, "ltn_adaptive", run_ltn_adaptive),
    ]

    results = {}
    for enabled, name, run in experiments:
        if enabled:
            results[name] = run(ctx)
    return results


if __name__ == "__main__":
    main()