'''
Experiments, losses and training loop.

    vanilla                  backbone trained with binary cross-entropy
    ltn_no_rules             LTN with the data axioms only:  Forall x+: P(x+),  Forall x-: Not(P(x-))
    ltn_no_pruning           LTN, data axioms and one axiom per rule, Forall x: condition(x) -> P(x)
                             (-> Not(P(x)) for the rules implying label 0), aggregated together
    ltn_no_pruning_weighted  the same, loss 1 - (DATA_WEIGHT * sat(data) + KNOWLEDGE_WEIGHT * sat(rules))
    ltn_pruning              weighted loss; after a calibration phase only the reliable rules stay (RulePruner)
    ltn_adaptive             weighted loss; per-rule weights driven by gradient alignment (AlignmentRuleWeights)

The rules are those of nesy/knowledge.py. Declare constraints (nesy/declare_knowledge.py, --declare) are never
connected to the label: with --declare in_p their truth values are extra inputs of P in every experiment,
P(x, C(x)); with --declare axioms the LTN experiments also aggregate the axioms Forall x: C(x) with the data
axioms (they involve no learnable predicate, so they do not change the gradients of P).

Protocol: each experiment starts from the same seed (same initialization and batch order); after every epoch the
model is scored on the validation set, the parameters with the best validation macro-F1 are kept, and training
stops after `patience` epochs without improvement (per dataset and experiment, see nesy/datasets.DATASETS).
'''
import copy
import itertools
import math
import os
import random
import statistics

import ltn
import numpy as np
import torch
import torch.nn.functional as F

from model.lstm import LSTMModel
from model.transformer import EventTransformer
from nesy.datasets import DATASETS, Training, make_loader
from nesy.evaluation import evaluate

EXPERIMENTS = ["vanilla", "ltn_no_rules", "ltn_no_pruning", "ltn_pruning", "ltn_no_pruning_weighted", "ltn_adaptive"]

# Weighted loss: 1 - (DATA_WEIGHT * sat(data axioms) + KNOWLEDGE_WEIGHT * sat(rule axioms))
DATA_WEIGHT = 0.8
KNOWLEDGE_WEIGHT = 0.2

# Adaptive rule weighting (gradient alignment)
ADAPTIVE_WARMUP_EPOCHS = 5         # uniform weights during warm-up
ADAPTIVE_ALIGN_LR = 1.0            # eta of the exponentiated-gradient update
ADAPTIVE_ALIGN_BATCHES = 8         # train batches used to estimate each rule's gradient
ADAPTIVE_HEAD_TENSORS = 2          # last N parameter tensors = classification head (weight + bias)
ADAPTIVE_FLOOR_START = 0.1         # uniform floor epsilon right after warm-up ...
ADAPTIVE_FLOOR_ANNEAL_EPOCHS = 15  # ... annealed linearly to 0 over this many epochs

Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
SatAgg = ltn.fuzzy_ops.SatAgg()


def set_seed(seed):
    '''Seed every source of randomness, and make cuDNN deterministic.'''
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class ModelInput:
    '''
    Input of P: the prefixes x, followed (--declare in_p) by the truth value of each Declare constraint repeated at
    every event, i.e. one extra feature per constraint in the feature-major layout, read by the models as a
    numerical feature.
    '''

    def __init__(self, layout, declare=None):
        self.seq_len = layout.seq_len
        self.declare = declare
        self.feature_names = list(layout.feature_names) + (declare.names if declare is not None else [])

    def __call__(self, x):
        if self.declare is None:
            return x
        truth = self.declare.truth(x).float()  # (B, K)
        return torch.cat([x, truth.unsqueeze(2).expand(-1, -1, self.seq_len).flatten(1)], dim=1)


def build_model(args, data, feature_names):
    seq_len = data.layout.seq_len
    if args.model == "lstm":
        return LSTMModel(feature_names, data.vocab_sizes, seq_len, hidden_size=args.hidden_size,
                         num_layers=args.num_layers, dropout=args.dropout)
    return EventTransformer(feature_names, data.vocab_sizes, seq_len, model_dim=args.model_dim,
                            num_layers=args.transformer_layers, num_heads=args.num_heads, dropout=args.dropout)


def indicator(rule):
    '''The antecedent of a rule as an ltn.Function with crisp 0/1 values.'''
    return ltn.Function(func=lambda x: rule.condition(x).float())


class Trainer:
    '''Model, predicate P, loaders and axioms of one experiment, built from the same seed for every experiment.'''

    def __init__(self, name, args, data, knowledge, declare, seed):
        self.name, self.args, self.data, self.knowledge, self.seed = name, args, data, knowledge, seed
        self.device = args.device
        self.training = DATASETS[data.name].training.get(name, Training())
        self.model_input = ModelInput(data.layout, declare if args.declare == "in_p" else None)
        self.declare_axioms = declare if args.declare == "axioms" else None
        self.rules = knowledge.rules
        self.antecedents = [indicator(rule) for rule in self.rules]
        set_seed(seed)
        self.model = build_model(args, data, self.model_input.feature_names).to(self.device)
        self.P = ltn.Predicate(self.model)
        self.train_loader = make_loader(data.train, args.batch_size, shuffle=True, seed=seed)
        self.val_loader = make_loader(data.val, args.batch_size)
        self.test_loader = make_loader(data.test, args.batch_size)

    ### AXIOMS (x: prefixes, x_in: input of P)

    def data_axioms(self, x_in, y):
        '''P holds on the positive prefixes and not on the negative ones.'''
        axioms = []
        if (y == 1).any():
            x_pos = ltn.Variable("x_pos", x_in[y == 1])
            axioms.append(Forall(x_pos, self.P(x_pos)))
        if (y == 0).any():
            x_neg = ltn.Variable("x_neg", x_in[y == 0])
            axioms.append(Forall(x_neg, Not(self.P(x_neg))))
        return axioms

    def base_axioms(self, x, x_in, y):
        '''Data axioms, and the Declare axioms Forall x: C(x) with --declare axioms.'''
        axioms = self.data_axioms(x_in, y)
        if self.declare_axioms is not None:
            axioms += self.declare_axioms.axioms(ltn.Variable("x_trace", x))
        return axioms

    def rule_axioms(self, x_all, indices=None):
        '''Forall x: condition(x) -> P(x), or -> Not(P(x)) for the rules implying label 0.'''
        indices = range(len(self.rules)) if indices is None else indices
        P_all = self.P(x_all)
        return [Forall(x_all, Implies(self.antecedents[k](x_all), P_all if self.rules[k].label == 1 else Not(P_all)))
                for k in indices]

    ### TRAINING

    def validate(self):
        return evaluate(self.model, self.val_loader, self.knowledge, self.model_input, self.device)["f1"]

    def run(self, loss_fn, epochs, on_epoch_end=None):
        '''Train with loss_fn(epoch, x, y), keep the best validation-F1 parameters, return the test metrics.'''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.training.lr)
        fit(self.model, loss_fn, optimizer, self.train_loader, self.validate, epochs, self.training.patience,
            self.device, self.name, on_epoch_end)
        return evaluate(self.model, self.test_loader, self.knowledge, self.model_input, self.device)


def fit(model, loss_fn, optimizer, train_loader, validate, epochs, patience, device, name, on_epoch_end=None):
    '''
    Minimize loss_fn(epoch, x, y) for at most `epochs` epochs. After each epoch the model is scored with validate()
    (validation macro-F1) and on_epoch_end(epoch) is called; the best parameters are restored at the end, and
    training stops after `patience` epochs without improvement (never if patience is None).
    '''
    best_score, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(epoch, x, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        score = validate()
        improved = score > best_score
        print(f"   [{name}] epoch {epoch + 1}/{epochs} | train loss {statistics.mean(losses):.4f} | "
              f"val F1 {score:.4f}{' *' if improved else ''}")
        if improved:
            best_score, best_state, bad_epochs = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"   [{name}] early stopping: no improvement for {patience} epochs")
                break
        if on_epoch_end is not None:
            on_epoch_end(epoch)
    model.load_state_dict(best_state)
    return best_score


def mixed_loss(data_sat, knowledge_sat):
    return 1 - (DATA_WEIGHT * data_sat + KNOWLEDGE_WEIGHT * knowledge_sat)


def weighted_p_mean_error(satisfactions, weights, p=2):
    '''Weighted p-mean-error aggregator: 1 - (sum_i w_i * (1 - s_i)^p)^(1/p).'''
    return 1 - (weights * (1 - satisfactions) ** p).sum() ** (1 / p)


### EXPERIMENTS: (trainer) -> (loss_fn(epoch, x, y), on_epoch_end or None)

def vanilla(t):
    def loss(epoch, x, y):
        return F.binary_cross_entropy(t.model(t.model_input(x)).squeeze(1), y)
    return loss, None


def ltn_no_rules(t):
    def loss(epoch, x, y):
        return 1 - SatAgg(*t.base_axioms(x, t.model_input(x), y))
    return loss, None


def ltn_no_pruning(t):
    def loss(epoch, x, y):
        x_in = t.model_input(x)
        return 1 - SatAgg(*t.base_axioms(x, x_in, y), *t.rule_axioms(ltn.Variable("x_all", x_in)))
    return loss, None


def ltn_no_pruning_weighted(t):
    def loss(epoch, x, y):
        x_in = t.model_input(x)
        return mixed_loss(SatAgg(*t.base_axioms(x, x_in, y)), SatAgg(*t.rule_axioms(ltn.Variable("x_all", x_in))))
    return loss, None


def rule_reliability(antecedent, consequent, variance_penalty, fired_only, no_support_score):
    '''
    Score in [0, 1] of how well `antecedent -> consequent` holds on the recorded samples:
    mean(implication) * exp(-variance_penalty * var(implication)). With fired_only, only the samples where the
    antecedent is satisfied count (no_support_score if there are fewer than 2); otherwise all samples count, so
    rarely-firing rules score close to 1.
    '''
    implication = Implies(antecedent, consequent).value
    if fired_only:
        fired = antecedent.value > 0.5
        if fired.sum() < 2:  # not enough evidence to estimate mean and variance
            return no_support_score
        implication = implication[fired]
    score = implication.mean().item() * math.exp(-variance_penalty * implication.var().item())
    return max(0.0, min(1.0, score))


class RulePruner:
    '''
    Drops the rules that look unreliable after a calibration phase (settings: nesy/datasets.Pruning).
      - epochs 0 .. gate_epoch: every rule is active
      - during stats_epoch: record the antecedent of each rule and P(x)
      - at the end of gate_epoch: score each rule against P (Not(P) for the rules implying label 0); afterwards
        only the rules scoring above the threshold stay active
    '''

    def __init__(self, trainer, settings):
        self.trainer, self.settings = trainer, settings
        self.gates = [0.0] * len(trainer.rules)
        self._antecedents = [[] for _ in trainer.rules]
        self._p = []

    def active_rules(self, epoch):
        if epoch <= self.settings.gate_epoch:
            return list(range(len(self.trainer.rules)))
        return [k for k, gate in enumerate(self.gates) if gate > self.settings.threshold]

    def observe(self, epoch, x_all):
        if epoch != self.settings.stats_epoch:
            return
        with torch.no_grad():
            for store, antecedent in zip(self._antecedents, self.trainer.antecedents):
                store.append(antecedent(x_all).value)
            self._p.append(self.trainer.P(x_all).value)

    def update_gates(self, epoch):
        if epoch != self.settings.gate_epoch:
            return
        s = self.settings
        with torch.no_grad():
            p = torch.cat(self._p)
            print(f"   Rule reliability after epoch {epoch + 1} (threshold {s.threshold}):")
            for k, (rule, values) in enumerate(zip(self.trainer.rules, self._antecedents)):
                consequent = ltn.LTNObject(p if rule.label == 1 else 1 - p, ["x_all"])
                antecedent = ltn.LTNObject(torch.cat(values), ["x_all"])
                self.gates[k] = rule_reliability(antecedent, consequent, s.variance_penalty, s.fired_only,
                                                 s.no_support_score)
                print(f"     {rule.name}: {self.gates[k]:.4f}{'' if self.gates[k] > s.threshold else '  (pruned)'}")


def ltn_pruning(t):
    pruner = RulePruner(t, DATASETS[t.data.name].pruning)

    def loss(epoch, x, y):
        x_in = t.model_input(x)
        x_all = ltn.Variable("x_all", x_in)
        data_sat = SatAgg(*t.base_axioms(x, x_in, y))
        pruner.observe(epoch, x_all)
        active = pruner.active_rules(epoch)
        if not active:
            return 1 - data_sat
        return mixed_loss(data_sat, SatAgg(*t.rule_axioms(x_all, active)))
    return loss, pruner.update_gates


def _flat_grad(loss, params, retain_graph=False):
    '''Gradient of `loss` w.r.t. `params`, flattened into one vector (zeros for unused params).'''
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(grads, params)])


class AlignmentRuleWeights:
    '''
    Per-rule weights driven by usefulness. Once per epoch (after warm-up), on the head parameters only,
        c_r = cos(gradient of the rule-r loss on train batches, gradient of the data loss on the validation set)
    and an exponentiated-gradient step on per-rule logits: logits_r += eta * c_r. Rules whose gradient points
    against the validation data gradient decay towards zero, helpful rules grow. The uniform floor is annealed to
    0, so a harmful rule can end up effectively pruned. The weights sum to one and are computed outside the
    training graph.
    '''

    def __init__(self, trainer):
        self.t = trainer
        n = len(trainer.rules)
        self.logits = torch.zeros(n, device=trainer.device)
        self.weights = torch.full((n,), 1 / n, device=trainer.device)

    def _floor(self, epoch):
        steps_since_warmup = epoch + 1 - ADAPTIVE_WARMUP_EPOCHS  # 0 at the first update
        return ADAPTIVE_FLOOR_START * max(0.0, 1 - steps_since_warmup / ADAPTIVE_FLOOR_ANNEAL_EPOCHS)

    def _data_gradient(self, params):
        total = torch.zeros(sum(p.numel() for p in params), device=self.t.device)
        for x, y in self.t.val_loader:
            x, y = x.to(self.t.device), y.to(self.t.device)
            total += _flat_grad(1 - SatAgg(*self.t.data_axioms(self.t.model_input(x), y)), params)
        return total

    def _rule_gradients(self, params):
        n_params = sum(p.numel() for p in params)
        totals = [torch.zeros(n_params, device=self.t.device) for _ in self.t.rules]
        for x, _ in itertools.islice(self.t.train_loader, ADAPTIVE_ALIGN_BATCHES):
            x_all = ltn.Variable("x_all", self.t.model_input(x.to(self.t.device)))
            for k, axiom in enumerate(self.t.rule_axioms(x_all)):  # the axioms share P(x_all)
                totals[k] += _flat_grad(1 - axiom.value, params, retain_graph=True)
        return totals

    def update(self, epoch):
        if epoch + 1 < ADAPTIVE_WARMUP_EPOCHS:
            return
        model = self.t.model
        params = list(model.parameters())[-ADAPTIVE_HEAD_TENSORS:]
        model.eval()  # no dropout noise in the gradients
        # cuDNN RNNs refuse to backpropagate in eval mode, so cuDNN is disabled for this step (LSTM backbone)
        with torch.backends.cudnn.flags(enabled=False):
            g_data = self._data_gradient(params)
            g_rules = self._rule_gradients(params)
        # A rule that never fires in the sampled batches has a zero gradient -> cosine 0 (neutral)
        alignment = torch.stack([F.cosine_similarity(g, g_data, dim=0, eps=1e-12) for g in g_rules])
        self.logits += ADAPTIVE_ALIGN_LR * alignment
        floor = self._floor(epoch)
        self.weights = floor / len(self.t.rules) + (1 - floor) * torch.softmax(self.logits, dim=0)
        print("   rule alignment:", [round(v, 3) for v in alignment.cpu().tolist()])
        print("   rule weights:  ", [round(v, 3) for v in self.weights.cpu().tolist()])


def ltn_adaptive(t):
    adaptive = AlignmentRuleWeights(t)

    def loss(epoch, x, y):
        x_in = t.model_input(x)
        rule_sats = torch.stack([a.value for a in t.rule_axioms(ltn.Variable("x_all", x_in))])
        return mixed_loss(SatAgg(*t.base_axioms(x, x_in, y)), weighted_p_mean_error(rule_sats, adaptive.weights))
    return loss, adaptive.update


BUILDERS = {"vanilla": vanilla, "ltn_no_rules": ltn_no_rules, "ltn_no_pruning": ltn_no_pruning,
            "ltn_pruning": ltn_pruning, "ltn_no_pruning_weighted": ltn_no_pruning_weighted,
            "ltn_adaptive": ltn_adaptive}


def run_experiment(name, args, data, knowledge, declare, seed):
    '''Train one experiment from `seed` and return its test metrics.'''
    trainer = Trainer(name, args, data, knowledge, declare, seed)
    loss_fn, on_epoch_end = BUILDERS[name](trainer)
    epochs = args.epochs if name == "vanilla" else args.epochs_ltn
    return trainer.run(loss_fn, epochs, on_epoch_end)
