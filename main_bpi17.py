import argparse
import itertools
import json
import math
import statistics
import warnings
from collections import Counter, namedtuple
from dataclasses import dataclass

import ltn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from data import preprocess_bpi17
from data.dataset import ModelConfig, NeSyDataset
from metrics import compute_metrics, compute_metrics_fa
from model.lstm import LSTMModel
from model.transformer import EventTransformer

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATASET = "bpi17"
DATA_PATH = "data_processed/bpi17.csv"
BATCH_SIZE = 32
SEQUENCE_LENGTH = 20

# Loss mixing: loss = 1 - (DATA_WEIGHT * data_sat + KNOWLEDGE_WEIGHT * knowledge_sat)
DATA_WEIGHT = 0.8
KNOWLEDGE_WEIGHT = 0.2

# Learning rate of the rule-based variants. None -> config.learning_rate (as in the original BPI17 script).
RULES_LEARNING_RATE = None

# Early-stopping patience (epochs without validation-F1 improvement)
VANILLA_PATIENCE = 5
PRUNING_PATIENCE = 5
WEIGHTED_PATIENCE = 5
ADAPTIVE_PATIENCE = 15

# Checkpoint files (one per experiment, prefixed by dataset)
VANILLA_CHECKPOINT = "bpi17_vanilla.pth"
NO_RULES_CHECKPOINT = "bpi17_ltn_no_rules.pth"
NO_PRUNING_CHECKPOINT = "bpi17_ltn_no_pruning.pth"
PRUNING_CHECKPOINT = "bpi17_ltn_pruning.pth"
WEIGHTED_CHECKPOINT = "bpi17_ltn_no_pruning_weighted.pth"
ADAPTIVE_CHECKPOINT = "bpi17_ltn_adaptive_align.pth"

# Rule pruning
PRUNING_CALIBRATION_EPOCH = 0   # rule statistics are collected in this epoch; pruning starts after it
PRUNING_GATE_THRESHOLD = 0.4
PRUNING_VARIANCE_PENALTY = 2.0

# Adaptive rule weighting (gradient-alignment version)
ADAPTIVE_WARMUP_EPOCHS = 5            # uniform weights during warm-up (E_p)
ADAPTIVE_ALIGN_LR = 1.0               # eta of the exponentiated-gradient update
ADAPTIVE_ALIGN_BATCHES = 8            # train batches used to estimate each rule's gradient
ADAPTIVE_HEAD_TENSORS = 2             # last N parameter tensors = classification head (weight + bias)
ADAPTIVE_FLOOR_START = 0.1            # uniform floor epsilon right after warm-up ...
ADAPTIVE_FLOOR_ANNEAL_EPOCHS = 15     # ... annealed linearly to 0 over this many epochs
ADAPTIVE_LEARNING_RATE = RULES_LEARNING_RATE   # same LR as the pruning variant, for a fair comparison

# Mode passed to `compute_metrics` for the final test evaluation (kept as in the original script).
TEST_MODE = "ltn"
ADAPTIVE_TEST_MODE = "ltn_w_k"


def block(index):
    """Columns of the `index`-th feature: each feature spans SEQUENCE_LENGTH consecutive columns."""
    return slice(index * SEQUENCE_LENGTH, (index + 1) * SEQUENCE_LENGTH)


# Feature blocks used by the rules (indices in the preprocessed layout; see `feature_names`).
# Rename these to the actual feature names.
SEQ_0 = block(0)
SEQ_1 = block(1)
SEQ_2 = block(2)
SEQ_6 = block(6)
REQUESTED_AMOUNT_COL = 140     # single column holding the (scaled) requested amount

TOKEN_IN_SEQ_0 = 11
TOKEN_A_IN_SEQ_1 = 6
TOKEN_B_IN_SEQ_1 = 3
TOKEN_IN_SEQ_6 = 2

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
    parser.add_argument("--results_path", type=str, default=None,
                        help="If set, dump the metrics of all experiments to this JSON file")
    return parser.parse_args()


# `positive` is True for rules `antecedent -> P` and False for rules `antecedent -> not P`.
Rule = namedtuple("Rule", ["name", "antecedent", "positive"])


def indicator(fn):
    """Wrap a boolean-valued function of the batch as an ltn.Function returning 0./1. floats."""
    return ltn.Function(func=lambda x: fn(x).float())


def build_rules(scalers):
    """Background-knowledge rules `antecedent(x) -> P(x)` or `antecedent(x) -> not P(x)`."""
    amount_threshold = scalers["case:RequestedAmount"].transform([[20000]])[0][0]

    credit_score_req_amount_less_20k = indicator(
        lambda x: (x[:, REQUESTED_AMOUNT_COL] < amount_threshold)
        & (x[:, SEQ_0] == TOKEN_IN_SEQ_0).any(dim=1)
        & (x[:, SEQ_2] != 0).any(dim=1))
    no_credit_score_greater_0 = indicator(
        lambda x: (x[:, SEQ_2] == 0).all(dim=1)
        & (x[:, SEQ_0] == TOKEN_IN_SEQ_0).any(dim=1))
    loan_takeover_req_amount_greater_20k = indicator(
        lambda x: (x[:, REQUESTED_AMOUNT_COL] > amount_threshold)
        & (x[:, SEQ_1] == TOKEN_A_IN_SEQ_1).any(dim=1))
    ex_loan_takeover = indicator(lambda x: (x[:, SEQ_1] == TOKEN_B_IN_SEQ_1).any(dim=1))
    application_type_new_credit = indicator(lambda x: (x[:, SEQ_6] == TOKEN_IN_SEQ_6).any(dim=1))

    return [
        Rule("credit_score_req_amount_less_20k", credit_score_req_amount_less_20k, True),
        Rule("no_credit_score_greater_0", no_credit_score_greater_0, False),
        Rule("loan_takeover_req_amount_greater_20k", loan_takeover_req_amount_greater_20k, False),
        Rule("ex_loan_takeover", ex_loan_takeover, True),
        Rule("application_type_new_credit", application_type_new_credit, True),
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
    event_log = pd.read_csv(DATA_PATH, dtype={"org:resource": str})
    splits, vocab_sizes, scalers = preprocess_bpi17.preprocess_eventlog(event_log, args.seed, args.setting)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = splits

    print("--- Label distribution")
    for split_name, labels in (("Training", y_train), ("Validation", y_val), ("Test", y_test)):
        print(f"--- {split_name} set")
        print(Counter(labels))
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
    optimizer = torch.optim.Adam(model.parameters(), lr=ctx.config.learning_rate)
    criterion = torch.nn.BCELoss()
    checkpoint = BestCheckpoint(model, VANILLA_PATIENCE, VANILLA_CHECKPOINT)

    for epoch in range(ctx.config.num_epochs):
        model.train()
        train_loss = train_one_epoch(model, ctx.train_loader, criterion, optimizer, ctx.device)
        print(f"Epoch {epoch + 1}/{ctx.config.num_epochs}, Loss: {train_loss}")

        val_loss, y_true, y_pred = evaluate_loader(model, ctx.val_loader, criterion, ctx.device)
        print(f"Validation Loss: {val_loss}")
        if checkpoint.update(f1_score(y_true, y_pred, average="macro")):
            print("Early stopping triggered")
            break

    checkpoint.restore()
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
    """Knowledge axioms: antecedent(x) -> P(x), or antecedent(x) -> not P(x) for negative rules."""
    formulas = []
    for rule in rules:
        consequent = P(x_all) if rule.positive else Not(P(x_all))
        formulas.append(Forall(x_all, Implies(rule.antecedent(x_all), consequent)))
    return formulas


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


def train_ltn(ctx, model, batch_loss, *, learning_rate=None, patience=None,
              checkpoint_path=None, on_epoch_end=None):
    """Generic LTN training loop.

    batch_loss(P, epoch, x, y) -> scalar loss tensor for one batch.
    checkpoint_path            -> if given, validate every epoch, keep the best-F1 weights
                                  (with early stopping after `patience`, if not None) and restore them
                                  at the end; if None, train for all epochs and keep the final weights.
    on_epoch_end(epoch, model) -> optional hook run after validation.
    """
    P = ltn.Predicate(model).to(ctx.device)
    optimizer = torch.optim.Adam(P.parameters(), lr=learning_rate or ctx.config.learning_rate)
    checkpoint = BestCheckpoint(model, patience, checkpoint_path) if checkpoint_path else None

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

        if checkpoint is not None:
            val_f1 = ctx.validation_f1(model)
            print("Validation F1 Score:", val_f1)
            if checkpoint.update(val_f1):
                print("Early stopping triggered")
                break
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)

    if checkpoint is not None:
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

    # Best-F1 checkpoint without early stopping (patience=None), as in the original script.
    return run_ltn(ctx, "LTN no rules", TEST_MODE, batch_loss,
                   learning_rate=RULES_LEARNING_RATE, checkpoint_path=NO_RULES_CHECKPOINT)


def run_ltn_no_pruning(ctx):
    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        formulas = supervised_formulas(P, x, y) + rule_formulas(ctx.rules, P, x_all)
        return 1 - SatAgg(*formulas)

    return run_ltn(ctx, "LTN w all rules", TEST_MODE, batch_loss,
                   learning_rate=RULES_LEARNING_RATE, checkpoint_path=NO_PRUNING_CHECKPOINT)


def run_ltn_no_pruning_weighted(ctx):
    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        knowledge_sat = SatAgg(*rule_formulas(ctx.rules, P, x_all))
        return mixed_loss(data_sat, knowledge_sat)

    return run_ltn(ctx, "LTN w weighted loss but no pruning", TEST_MODE, batch_loss,
                   learning_rate=RULES_LEARNING_RATE,
                   patience=WEIGHTED_PATIENCE, checkpoint_path=WEIGHTED_CHECKPOINT)


class RulePruner:
    """Drops rules that look unreliable after a calibration epoch.

    * Epochs 0..calibration_epoch: every rule is active.
    * During `calibration_epoch`: record each rule's antecedent truth values and P(x).
    * At the end of `calibration_epoch`: score each rule (on the samples where its antecedent
      fires, against P or not-P depending on the rule's polarity); afterwards only rules whose
      score exceeds `threshold` stay active.
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
        return [rule for rule, gate in zip(self.rules, self.gates) if gate > self.threshold]

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
            p_value = torch.cat(self._p_truth)
            print(f"Gating scores after epoch {epoch}:")
            for i, (rule, truth) in enumerate(zip(self.rules, self._antecedent_truth)):
                antecedent = ltn.LTNObject(torch.cat(truth), ["x_All"])
                consequent = ltn.LTNObject(p_value if rule.positive else 1 - p_value, ["x_All"])
                self.gates[i] = rule_reliability(
                    antecedent, consequent, PRUNING_VARIANCE_PENALTY, fired_only=True)
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

    def on_epoch_end(epoch, model):
        pruner.update_gates(epoch)
        # Diagnostics only: the test score is printed each epoch but never used for selection.
        with torch.no_grad():
            model.eval()
            _, f1, _, _, _ = compute_metrics_fa(
                ctx.test_loader, model, ctx.device, "nesy", ctx.scalers, DATASET)
        print("Test F1 Score:", f1)

    return run_ltn(ctx, "LTN w pruning", TEST_MODE, batch_loss,
                   learning_rate=RULES_LEARNING_RATE, patience=PRUNING_PATIENCE,
                   checkpoint_path=PRUNING_CHECKPOINT, on_epoch_end=on_epoch_end)


# --------------------------------------------------------------------------- #
# Adaptive rule weighting via gradient alignment
# --------------------------------------------------------------------------- #

def _flat_grad(loss, params):
    """Gradient of `loss` w.r.t. `params`, flattened into one vector (zeros for unused params)."""
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, p in zip(grads, params)])


class AlignmentRuleWeights:
    """Per-rule weights driven by *usefulness*, not consistency.

    Once per epoch (after warm-up) we measure, on the head parameters only,

        c_r = cos( grad of rule-r loss on train batches ,  grad of data loss on validation set )

    and run an exponentiated-gradient step on per-rule logits:  logits_r += eta * c_r.
    Rules whose gradient points against the validation data gradient (c_r < 0) decay
    towards zero; helpful rules grow.  The uniform floor epsilon is annealed to 0, so a
    harmful rule can end up effectively pruned (soft version of the two-stage pruning).
    Weights always sum to one and are computed outside the training graph.
    """

    def __init__(self, ctx):
        self.ctx = ctx
        self.rules = ctx.rules
        n_rules = len(self.rules)
        self.logits = torch.zeros(n_rules, device=ctx.device)
        self.weights = torch.full((n_rules,), 1 / n_rules, device=ctx.device)

    def _floor(self, epoch):
        steps_since_warmup = epoch + 1 - ADAPTIVE_WARMUP_EPOCHS      # 0 at the first update
        return ADAPTIVE_FLOOR_START * max(0.0, 1 - steps_since_warmup / ADAPTIVE_FLOOR_ANNEAL_EPOCHS)

    def _data_gradient(self, P, params):
        total = torch.zeros(sum(p.numel() for p in params), device=self.ctx.device)
        for x, y in self.ctx.val_loader:
            loss = 1 - SatAgg(*supervised_formulas(P, x.to(self.ctx.device), y))
            total += _flat_grad(loss, params)
        return total

    def _rule_gradients(self, P, params):
        n_params = sum(p.numel() for p in params)
        totals = [torch.zeros(n_params, device=self.ctx.device) for _ in self.rules]
        for x, _ in itertools.islice(self.ctx.train_loader, ADAPTIVE_ALIGN_BATCHES):
            x_all = ltn.Variable("x_All", x.to(self.ctx.device))
            for i, formula in enumerate(rule_formulas(self.rules, P, x_all)):
                totals[i] += _flat_grad(1 - formula.value, params)
        return totals

    def update(self, epoch, model):
        if epoch + 1 < ADAPTIVE_WARMUP_EPOCHS:
            return
        P = ltn.Predicate(model).to(self.ctx.device)
        params = list(model.parameters())[-ADAPTIVE_HEAD_TENSORS:]

        model.eval()  # no dropout noise in the gradients
        # cuDNN RNNs refuse to backprop in eval mode, so disable cuDNN for this step (LSTM backbone).
        with torch.backends.cudnn.flags(enabled=False):
            g_data = self._data_gradient(P, params)
            g_rules = self._rule_gradients(P, params)

        # A rule that never fires in the sampled batches has a zero gradient -> cosine 0 (neutral).
        alignment = torch.stack([F.cosine_similarity(g, g_data, dim=0, eps=1e-12) for g in g_rules])

        self.logits += ADAPTIVE_ALIGN_LR * alignment
        floor = self._floor(epoch)
        n_rules = len(self.rules)
        self.weights = floor / n_rules + (1 - floor) * torch.softmax(self.logits, dim=0)

        print("Rule alignment:", [round(v, 3) for v in alignment.cpu().tolist()])
        print("Rule weights:  ", [round(v, 3) for v in self.weights.cpu().tolist()])


def run_ltn_adaptive(ctx):
    adaptive = AlignmentRuleWeights(ctx)

    def batch_loss(P, epoch, x, y):
        x_all = ltn.Variable("x_All", x)
        data_sat = SatAgg(*supervised_formulas(P, x, y))
        rule_sats = torch.stack([f.value for f in rule_formulas(ctx.rules, P, x_all)])
        knowledge_sat = weighted_p_mean_error(rule_sats, adaptive.weights)
        return mixed_loss(data_sat, knowledge_sat)

    return run_ltn(ctx, "LTN w adaptive rule weights (gradient alignment)", ADAPTIVE_TEST_MODE, batch_loss,
                   learning_rate=ADAPTIVE_LEARNING_RATE,
                   patience=ADAPTIVE_PATIENCE, checkpoint_path=ADAPTIVE_CHECKPOINT,
                   on_epoch_end=adaptive.update)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    args = get_args()
    print("Arguments:", args)
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

    if args.results_path:
        with open(args.results_path, "w") as f:
            json.dump(results, f, default=float)
    return results


if __name__ == "__main__":
    main()