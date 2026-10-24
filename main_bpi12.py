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

from data import preprocess_bpi12
from data.dataset import ModelConfig, NeSyDataset
from metrics import compute_metrics
from model.lstm import LSTMModel
from model.transformer import EventTransformer

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATASET = "bpi12"
BATCH_SIZE = 32
SEQUENCE_LENGTH = 40
BEST_MODEL_PATH = "best_model.pth"

# Each input row is [resources | activities | amounts], each block SEQUENCE_LENGTH long.
RESOURCES = slice(0, SEQUENCE_LENGTH)
ACTIVITIES = slice(SEQUENCE_LENGTH, 2 * SEQUENCE_LENGTH)
AMOUNTS = slice(2 * SEQUENCE_LENGTH, 3 * SEQUENCE_LENGTH)

# Encoded token ids used by the background-knowledge rules.
RESOURCE_ID_11169 = 48
RESOURCE_ID_10910 = 21
ACTIVITY_ID_O_CANCELLED = 11
ACTIVITY_ID_O_SENT_BACK = 15
MIN_CANCELLATIONS = 3

# Loss mixing: loss = 1 - (DATA_WEIGHT * data_sat + KNOWLEDGE_WEIGHT * knowledge_sat)
DATA_WEIGHT = 0.8
KNOWLEDGE_WEIGHT = 0.2

# Rule pruning
PRUNING_WARMUP_EPOCHS = 5     # all rules active up to (and including) this epoch
PRUNING_STATS_EPOCH = 1       # epoch during which rule statistics are collected
PRUNING_GATE_THRESHOLD = 0.5
PRUNING_PATIENCE = 8

# Adaptive rule weighting
ADAPTIVE_PATIENCE = 15
ADAPTIVE_EMA_DECAY = 0.8
ADAPTIVE_UNIFORM_MIX = 0.1    # share of the weights that stays uniform
ADAPTIVE_VARIANCE_PENALTY = 2.0
PRUNING_VARIANCE_PENALTY = 1.0

WEIGHTED_PATIENCE = 15

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
                        help="Train LTN model with adaptive rule weights and weighted loss")
    parser.add_argument("--setting", type=str, default="compliance",
                        help="Setting for the experiment (compliance or temporal)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


Rule = namedtuple("Rule", ["name", "antecedent"])


def build_rules(scalers):
    """Background-knowledge rules of the form `antecedent(x) -> not P(x)`.

    P is "the case is accepted"; every rule describes a pattern that implies rejection.
    """
    amount_scaler = scalers["case:AMOUNT_REQ"]

    def scaled(amount):
        return amount_scaler.transform([[amount]])[0][0]

    t_10k, t_50k, t_60k = scaled(10_000), scaled(50_000), scaled(60_000)

    # `> 0` skips padded positions.
    amount_below_10k = ltn.Function(
        func=lambda x: ((x[:, AMOUNTS] > 0) & (x[:, AMOUNTS] < t_10k)).any(dim=1))
    amount_above_50k = ltn.Function(
        func=lambda x: (x[:, AMOUNTS] > t_50k).any(dim=1))
    amount_below_60k = ltn.Function(
        func=lambda x: ((x[:, AMOUNTS] > 0) & (x[:, AMOUNTS] < t_60k)).any(dim=1))
    has_resource_11169 = ltn.Function(
        func=lambda x: (x[:, RESOURCES] == RESOURCE_ID_11169).any(dim=1))
    has_resource_10910 = ltn.Function(
        func=lambda x: (x[:, RESOURCES] == RESOURCE_ID_10910).any(dim=1))
    o_cancelled_repeatedly = ltn.Function(
        func=lambda x: (x[:, ACTIVITIES] == ACTIVITY_ID_O_CANCELLED).sum(dim=1) >= MIN_CANCELLATIONS)
    o_sent_back = ltn.Function(
        func=lambda x: (x[:, ACTIVITIES] == ACTIVITY_ID_O_SENT_BACK).any(dim=1))

    return [
        Rule("amount_below_10k", amount_below_10k),
        Rule("amount_above_50k_and_below_60k", lambda x: And(amount_above_50k(x), amount_below_60k(x))),
        Rule("resource_11169", has_resource_11169),
        Rule("resource_10910", has_resource_10910),
        Rule("o_cancelled_repeatedly", o_cancelled_repeatedly),
        Rule("o_sent_back", o_sent_back),
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
        dataset=DATASET,
        sequence_length=SEQUENCE_LENGTH,
        seed=args.seed,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("-- Reading dataset")
    event_log = pd.read_csv(f"data_processed/{DATASET}.csv", dtype={"org:resource": str})
    splits, vocab_sizes, scalers = preprocess_bpi12.preprocess_eventlog(event_log, args.seed, args.setting)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = splits

    print("--- Label distribution")
    print("--- Training set")
    print(Counter(y_train))
    print("--- Test set")
    print(Counter(y_test))

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


# --------------------------------------------------------------------------- #
# Vanilla (non-LTN) baseline
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader` (trains if an optimizer is given). Returns the mean loss."""
    losses = []
    for x, y in loader:
        output = model(x.to(device))
        loss = criterion(output.squeeze(1).cpu(), y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return statistics.mean(losses)


def run_vanilla(ctx):
    model = ctx.build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=ctx.config.learning_rate)
    criterion = torch.nn.BCELoss()

    val_losses = []
    for epoch in range(ctx.config.num_epochs):
        model.train()
        train_loss = run_epoch(model, ctx.train_loader, criterion, ctx.device, optimizer)
        print(f"Epoch {epoch + 1}/{ctx.config.num_epochs}, Loss: {train_loss}")

        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, ctx.val_loader, criterion, ctx.device)
        print(f"Validation Loss: {val_loss}")
        val_losses.append(val_loss)

        if epoch >= 5 and val_losses[-1] > val_losses[-2]:
            print("Validation loss increased, stopping training")
            break

    y_true, y_pred = predict(model, ctx.test_loader, ctx.device)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
    }
    report("vanilla", metrics)
    return metrics


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, predictions = [], []
    for x, y in loader:
        outputs = model(x.to(device)).cpu().numpy().flatten()
        predictions.append((outputs > 0.5).astype(float))
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(predictions)


# --------------------------------------------------------------------------- #
# Shared LTN building blocks
# --------------------------------------------------------------------------- #

class BestCheckpoint:
    """Saves the weights with the best validation F1 and tracks early-stopping patience."""

    def __init__(self, model, patience=None, path=BEST_MODEL_PATH):
        self.model = model
        self.patience = patience
        self.path = path
        self.best_f1 = 0.0
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
    """Knowledge axioms: for every rule, antecedent(x) -> not P(x)."""
    return [Forall(x_all, Implies(rule.antecedent(x_all), Not(P(x_all)))) for rule in rules]


def mixed_loss(data_sat, knowledge_sat):
    return 1 - (DATA_WEIGHT * data_sat + KNOWLEDGE_WEIGHT * knowledge_sat)


def rule_reliability(antecedent, consequent, variance_penalty, no_support_score):
    """Score in [0, 1] of how well `antecedent -> consequent` holds where the antecedent fires.

    score = mean(implication) * exp(-variance_penalty * var(implication)),
    computed only over samples where the antecedent is satisfied.
    """
    fired = antecedent.value > 0.5
    if fired.sum() < 2:  # not enough evidence to estimate mean and variance
        return no_support_score
    implication = Implies(antecedent, consequent).value[fired]
    score = implication.mean().item() * math.exp(-variance_penalty * implication.var().item())
    return max(0.0, min(1.0, score))


def weighted_p_mean_error(satisfactions, weights, p=2):
    """Weighted p-mean-error aggregator: 1 - (sum_i w_i * (1 - s_i)^p)^(1/p)."""
    weighted_errors = weights * (1 - satisfactions) ** p
    return 1 - weighted_errors.sum() ** (1 / p)


def train_ltn(ctx, model, batch_loss, *, patience=None, restore_best=True, on_epoch_end=None):
    """Generic LTN training loop.

    batch_loss(P, epoch, x, y) -> scalar loss tensor for one batch.
    on_epoch_end(epoch)        -> optional hook run after validation.
    """
    P = ltn.Predicate(model).to(ctx.device)
    optimizer = torch.optim.Adam(P.parameters(), lr=ctx.config.learning_rate)
    checkpoint = BestCheckpoint(model, patience)

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
            on_epoch_end(epoch)

    if restore_best:
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

    return run_ltn(ctx, "LTN w/o knowledge", "ltn", batch_loss)


def run_ltn_no_pruning(ctx):
    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        formulas = supervised_formulas(P, x, y) + rule_formulas(ctx.rules, P, x_all)
        return 1 - SatAgg(*formulas)

    return run_ltn(ctx, "LTN w/o rule pruning", "ltn_w_k", batch_loss)


def run_ltn_no_pruning_weighted(ctx):
    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        knowledge_sat = SatAgg(*rule_formulas(ctx.rules, P, x_all))
        return mixed_loss(data_sat, knowledge_sat)

    return run_ltn(ctx, "LTN w weighted loss but no pruning", "ltn_w_k", batch_loss,
                   patience=WEIGHTED_PATIENCE)


class RulePruner:
    """Drops rules that look unreliable after a warm-up phase.

    * Epochs 0..warmup_epochs: every rule is active.
    * During `stats_epoch`: record each rule's antecedent truth values and `not P(x)`.
    * At the end of `warmup_epochs`: score each rule; afterwards only rules whose
      score exceeds `threshold` stay active.
    """

    def __init__(self, rules, warmup_epochs, stats_epoch, threshold):
        self.rules = rules
        self.warmup_epochs = warmup_epochs
        self.stats_epoch = stats_epoch
        self.threshold = threshold
        self.gates = [0.0] * len(rules)
        self._antecedent_truth = [[] for _ in rules]
        self._not_p_truth = []

    def active_rules(self, epoch):
        if epoch <= self.warmup_epochs:
            return list(self.rules)
        return [rule for rule, gate in zip(self.rules, self.gates) if gate > self.threshold]

    def observe(self, P, epoch, x_all):
        if epoch != self.stats_epoch:
            return
        with torch.no_grad():
            for store, rule in zip(self._antecedent_truth, self.rules):
                store.append(rule.antecedent(x_all).value)
            self._not_p_truth.append(Not(P(x_all)).value)

    def update_gates(self, epoch):
        if epoch != self.warmup_epochs:
            return
        consequent = ltn.LTNObject(torch.cat(self._not_p_truth), ["x_All"])
        print(f"Gating scores after epoch {self.warmup_epochs}:")
        for i, (rule, truth) in enumerate(zip(self.rules, self._antecedent_truth)):
            antecedent = ltn.LTNObject(torch.cat(truth), ["x_All"])
            # A rule that never fired is kept (it cannot hurt training).
            self.gates[i] = rule_reliability(
                antecedent, consequent, PRUNING_VARIANCE_PENALTY, no_support_score=1.0)
            print(f"  {rule.name}: {self.gates[i]}")


def run_ltn_pruning(ctx):
    pruner = RulePruner(ctx.rules, PRUNING_WARMUP_EPOCHS, PRUNING_STATS_EPOCH, PRUNING_GATE_THRESHOLD)

    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        pruner.observe(P, epoch, x_all)
        active = pruner.active_rules(epoch)
        if not active:
            return 1 - data_sat
        return mixed_loss(data_sat, SatAgg(*rule_formulas(active, P, x_all)))

    return run_ltn(ctx, "LTN w pruning", "ltn_w_k", batch_loss,
                   patience=PRUNING_PATIENCE, restore_best=True,
                   on_epoch_end=pruner.update_gates)


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

    def update(self, epoch=None):
        with torch.no_grad():
            # NOTE: the consequent is P(x), as in the original script, although the rules
            # themselves conclude `not P(x)` (the pruning variant uses not P). Check intent.
            consequent = ltn.LTNObject(torch.cat(self._p_truth), ["x_All"])
            raw_scores = torch.tensor(
                [rule_reliability(ltn.LTNObject(torch.cat(truth), ["x_All"]), consequent,
                                  ADAPTIVE_VARIANCE_PENALTY, no_support_score=0.0)
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
                   patience=ADAPTIVE_PATIENCE, on_epoch_end=adaptive.update)


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