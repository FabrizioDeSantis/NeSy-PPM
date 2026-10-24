'''
Declare constraints as LTNtorch predicates.

The predicates operate on the flattened model input x of shape (B, num_features * seq_len)
built by create_ngrams: the activity sequence is sliced out of x, so the same ltn.Variable
used for the neural predicate P can be passed to the Declare predicates. Their truth values are
crisp (0/1), as for the logical rules of nesy/knowledge.py.

Two kinds of predicates are available:
  - one ltn.Predicate per mined constraint, with its activities bound:   kb["Response(A, B)"](x)
  - one ltn.Predicate per template, with activities given as ltn.Constant:
        Response = kb.template_predicate("response")
        Response(x, kb.constant("ER Triage"), kb.constant("ER Sepsis Triage"))

Example:
    kb = DeclareKB.from_json("results/declare_sepsis.json", feature_names, max_prefix_length)
    kb = kb.prune(min_confidence=0.95, max_ltn_gap=0.01, prefix_safe_only=True, top_k=20)
    x_All = ltn.Variable("x_All", x)
    formulas += kb.axioms(x_All, P, mode="implies")        # Forall x: C_k(x) -> P(x), for each k
    formulas += [Forall(x_All, Not(P(x_All)), cond_vars=[x_All], cond_fn=kb.cond_fn(c, satisfied=False))
                 for c in kb]                                # guarded: violating prefixes -> not P
'''
from dataclasses import dataclass

import ltn
import torch

from declare.templates import TEMPLATES, PAD
from declare.miner import load_constraints, save_constraints
from declare.selection import prune_constraints, sort_constraints, DEFAULT_SORT

CARDINALITY_TEMPLATES = ("existence", "absence", "exactly")


def _as_column(c, batch_size):
    # LTN expands a Constant of shape (1,) to (B, 1) and a 0-d Constant to (B,)
    c = torch.as_tensor(c)
    if c.dim() == 0:
        return c.view(1, 1)
    return c.reshape(batch_size, -1)[:, :1] if c.shape[0] == batch_size else c.reshape(-1)[:1].view(1, 1)


def _values(x):
    return x.value if isinstance(x, ltn.core.LTNObject) else x


def declare_function(template, act_offset, seq_len):
    '''
    Template-level function f(x, a[, b | n]) -> float tensor in {0, 1} of shape (B,), usable as
    `ltn.Predicate(func=...)` with activity codes passed as ltn.Constant.
    '''
    spec = TEMPLATES[template]

    def acts_of(x):
        return x[:, act_offset:act_offset + seq_len].round().long()

    if spec.arity == 1 and template in CARDINALITY_TEMPLATES:
        def func(x, a, n):
            traces = acts_of(x)
            mask_a = traces == _as_column(a, len(traces)).to(traces.device)
            n = _as_column(n, len(traces)).to(traces.device).squeeze(-1)
            return spec.fn(mask_a, traces != PAD, n).float()
    elif spec.arity == 1:
        def func(x, a):
            traces = acts_of(x)
            mask_a = traces == _as_column(a, len(traces)).to(traces.device)
            return spec.fn(mask_a, traces != PAD).float()
    else:
        def func(x, a, b):
            traces = acts_of(x)
            mask_a = traces == _as_column(a, len(traces)).to(traces.device)
            mask_b = traces == _as_column(b, len(traces)).to(traces.device)
            return spec.fn(mask_a, mask_b, traces != PAD).float()
    func.__name__ = spec.display_name
    return func


def constraint_function(constraint, act_offset, seq_len):
    '''Function f(x) -> float tensor in {0, 1} of shape (B,): satisfaction of one constraint.'''
    spec, a, b, n = constraint.spec, constraint.a, constraint.b, constraint.n

    def func(x):
        traces = x[:, act_offset:act_offset + seq_len].round().long()
        valid = traces != PAD
        if spec.arity == 1:
            return spec.fn(traces == a, valid, n).float()
        return spec.fn(traces == a, traces == b, valid).float()
    func.__name__ = constraint.name
    return func


def declare_predicate(template, act_offset, seq_len):
    return ltn.Predicate(func=declare_function(template, act_offset, seq_len))


def make_declare_predicates(feature_names, seq_len, templates=None, activity_feature="concept:name"):
    '''Dictionary template name -> template-level ltn.Predicate for the given model input layout.'''
    act_offset = feature_names.index(activity_feature) * seq_len
    return {t: declare_predicate(t, act_offset, seq_len) for t in (templates or TEMPLATES)}


@dataclass
class LabelRule:
    '''
    A Declare constraint turned into a rule on the label predicate P:
        mode "violation_implies_not":  not C(x) -> not P(x)
        mode "violation_implies":      not C(x) -> P(x)
    precision is the fraction of violating individuals with the implied label, violations their number
    (both measured on the data used to select the rule).
    '''
    constraint: object
    mode: str
    precision: float
    violations: int

    @property
    def implied_label(self):
        return 0 if self.mode == "violation_implies_not" else 1

    def __str__(self):
        head = "not P" if self.implied_label == 0 else "P"
        return (f"not {self.constraint.name} -> {head}  "
                f"[precision={self.precision:.3f} violations={self.violations} conf={self.constraint.confidence:.3f}]")


class DeclareKB:
    '''An ordered set of Declare constraints grounded as LTNtorch predicates on the model input.'''

    def __init__(self, constraints, feature_names, seq_len, vocab=None, activity_feature="concept:name"):
        self.constraints = list(constraints)
        self.feature_names = list(feature_names)
        self.vocab = vocab
        self.seq_len = seq_len
        self.activity_feature = activity_feature
        self.act_offset = self.feature_names.index(activity_feature) * seq_len
        self._template_predicates = {}
        self._predicates = {}

        # Default connectives, as in nesy/experiments.py
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

    @classmethod
    def from_json(cls, path, feature_names, seq_len, **kwargs):
        constraints, vocab = load_constraints(path)
        return cls(constraints, feature_names, seq_len, vocab=vocab, **kwargs)

    def save(self, path, **params):
        save_constraints(path, self.constraints, vocab=self.vocab, params=params)

    ### ORDERING AND PRUNING (return new KBs, predicates are shared)

    def _with_constraints(self, constraints):
        kb = DeclareKB.__new__(DeclareKB)
        kb.__dict__.update(self.__dict__)
        kb.constraints = list(constraints)
        return kb

    def sort(self, by=DEFAULT_SORT):
        '''New KB sorted by the given metrics (see declare.selection.METRICS).'''
        return self._with_constraints(sort_constraints(self.constraints, by))

    def prune(self, **kwargs):
        '''New KB pruned with declare.selection.prune_constraints(**kwargs).'''
        return self._with_constraints(prune_constraints(self.constraints, **kwargs))

    def filter(self, fn):
        '''New KB with the constraints for which fn(constraint) is True.'''
        return self._with_constraints([c for c in self.constraints if fn(c)])

    ### PREDICATES

    def predicate(self, constraint):
        '''ltn.Predicate P_c(x) for one constraint (object, name or index in the KB).'''
        if isinstance(constraint, int):
            constraint = self.constraints[constraint]
        elif isinstance(constraint, str):
            matches = [c for c in self.constraints if c.name == constraint]
            if not matches:
                raise KeyError(constraint)
            constraint = matches[0]
        if constraint.key not in self._predicates:
            self._predicates[constraint.key] = ltn.Predicate(
                func=constraint_function(constraint, self.act_offset, self.seq_len))
        return self._predicates[constraint.key]

    __getitem__ = predicate

    @property
    def predicates(self):
        '''Constraint name -> ltn.Predicate, in KB order.'''
        return {c.name: self.predicate(c) for c in self.constraints}

    def template_predicate(self, template):
        '''Template-level ltn.Predicate T(x, a[, b | n]) taking activities as ltn.Constant.'''
        if template not in self._template_predicates:
            self._template_predicates[template] = declare_predicate(template, self.act_offset, self.seq_len)
        return self._template_predicates[template]

    def constant(self, value, device=None):
        '''ltn.Constant for an activity (name or code) or a cardinality n.'''
        if isinstance(value, str):
            value = self.vocab[value]
        return ltn.Constant(torch.tensor([value], device=device))

    ### GROUNDINGS AND FORMULAS

    def ground_constraint(self, constraint, x_var):
        '''LTNObject with the truth value of one constraint for each individual of x_var.'''
        return self.predicate(constraint)(x_var)

    def ground(self, x_var):
        '''One LTNObject per constraint, in KB order.'''
        return [self.ground_constraint(c, x_var) for c in self.constraints]

    def axioms(self, x_var, P=None, mode="satisfy"):
        '''
        One LTN formula per constraint C, quantified over x_var:
            "satisfy":               Forall x: C(x)
            "implies":               Forall x: C(x) -> P(x)
            "implies_not":           Forall x: C(x) -> not P(x)
            "violation_implies":     Forall x: not C(x) -> P(x)
            "violation_implies_not": Forall x: not C(x) -> not P(x)
        or a callable (constraint, C(x), P(x)) -> LTNObject, quantified with Forall.
        "satisfy" does not involve P: it measures the knowledge satisfaction but gives no
        gradient to the model.
        '''
        if P is None and (callable(mode) or mode != "satisfy"):
            raise ValueError(f"mode {mode!r} needs the neural predicate P")
        Px = P(x_var) if P is not None else None
        return [self.Forall(x_var, self._build(mode)(c, C, Px)) for c, C in zip(self.constraints, self.ground(x_var))]

    def _build(self, mode):
        if callable(mode):
            return mode
        Not, Implies = self.Not, self.Implies
        return {
            "satisfy": lambda c, C, Px: C,
            "implies": lambda c, C, Px: Implies(C, Px),
            "implies_not": lambda c, C, Px: Implies(C, Not(Px)),
            "violation_implies": lambda c, C, Px: Implies(Not(C), Px),
            "violation_implies_not": lambda c, C, Px: Implies(Not(C), Not(Px)),
        }[mode]

    ### LABEL RULES

    def violation_stats(self, x, y, batch_size=4096):
        '''Per constraint: number of individuals of x violating it and how many of them have label 1.'''
        violations = torch.zeros(len(self), dtype=torch.long)
        positives = torch.zeros(len(self), dtype=torch.long)
        y = torch.as_tensor(y).round().long()
        for i in range(0, len(x), batch_size):
            violated = self.satisfaction(x[i:i + batch_size]).cpu() < 0.5
            labels = y[i:i + batch_size].cpu().unsqueeze(1)
            violations += violated.sum(dim=0)
            positives += (violated & (labels == 1)).sum(dim=0)
        return violations, positives

    def label_rules(self, x, y, min_precision=0.95, min_violations=10, top_k=None):
        '''
        Turn the constraints into label rules: if (almost) all the individuals of x violating a constraint
        have the same label y, the violation implies that label. Violations of prefix-safe constraints
        are permanent, so the rule also holds for every continuation of the prefix.
        Rules are kept if precision >= min_precision and violations >= min_violations, and sorted by
        precision, then number of violations.
        '''
        violations, positives = self.violation_stats(x, y)
        rules = []
        for c, n_viol, n_pos in zip(self.constraints, violations.tolist(), positives.tolist()):
            if n_viol < max(1, min_violations):
                continue
            frac_pos = n_pos / n_viol
            if 1 - frac_pos >= min_precision:
                rules.append(LabelRule(c, "violation_implies_not", 1 - frac_pos, n_viol))
            elif frac_pos >= min_precision:
                rules.append(LabelRule(c, "violation_implies", frac_pos, n_viol))
        rules.sort(key=lambda r: (-r.precision, -r.violations, r.constraint.name))
        return rules[:top_k] if top_k is not None else rules

    def rule_stats(self, rules, x, y):
        '''(precision, violations) of already selected label rules on other data (e.g. a validation set).'''
        kb = self._with_constraints([r.constraint for r in rules])
        violations, positives = kb.violation_stats(x, y)
        stats = []
        for r, n_viol, n_pos in zip(rules, violations.tolist(), positives.tolist()):
            hits = n_pos if r.implied_label == 1 else n_viol - n_pos
            stats.append((hits / n_viol if n_viol else None, n_viol))
        return stats

    def label_axioms(self, x_var, P, rules):
        '''One LTN formula per label rule: Forall x: not C(x) -> P(x) / not P(x).'''
        Px = P(x_var)
        return [self.Forall(x_var, self._build(r.mode)(r.constraint, self.ground_constraint(r.constraint, x_var), Px))
                for r in rules]

    def label_compliance(self, x, predictions, rules, y=None):
        '''
        Number of (rule, individual) pairs where the rule fires (the constraint is violated) and the
        prediction agrees with the implied label, and number of pairs where the rule fires.
        With y, only pairs whose true label agrees with the rule are counted, as in metrics.compute_metrics.
        '''
        if not rules:
            return 0, 0
        kb = self._with_constraints([r.constraint for r in rules])
        fired = kb.satisfaction(x) < 0.5
        implied = torch.tensor([r.implied_label for r in rules], device=fired.device).unsqueeze(0)
        predictions = torch.as_tensor(predictions, device=fired.device).round().long().view(-1, 1)
        if y is not None:
            fired &= torch.as_tensor(y, device=fired.device).round().long().view(-1, 1) == implied
        return int((fired & (predictions == implied)).sum()), int(fired.sum())

    def cond_fn(self, constraint, satisfied=True):
        '''
        Condition for LTN guarded quantification: restricts Forall to the individuals that satisfy
        (or violate) the constraint, e.g. Forall(x, Not(P(x)), cond_vars=[x], cond_fn=kb.cond_fn(c, False)).
        '''
        func = constraint_function(constraint, self.act_offset, self.seq_len)
        return lambda x: (func(_values(x)) > 0.5) == satisfied

    def satisfaction(self, x):
        '''Crisp (B, K) tensor: satisfaction of each constraint on the raw model input x.'''
        x = _values(x)
        if not self.constraints:
            return torch.zeros(len(x), 0, device=x.device)
        return torch.stack([constraint_function(c, self.act_offset, self.seq_len)(x)
                            for c in self.constraints], dim=1)

    def __len__(self):
        return len(self.constraints)

    def __iter__(self):
        return iter(self.constraints)

    def __repr__(self):
        lines = [f"DeclareKB({len(self)} constraints)"] + [f"  {c}" for c in self.constraints]
        return "\n".join(lines)
