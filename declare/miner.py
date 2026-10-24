'''
Discovery of Declare constraints from integer-encoded traces.

For every candidate constraint the miner computes, over the (weighted) traces:
    support            = fraction of traces satisfying the constraint (vacuous satisfaction included)
    activation_support = fraction of traces in which the constraint is activated
    confidence         = fraction of activated traces satisfying the constraint
and, when labels are given, the confidence restricted to positive / negative traces.
A constraint is kept if activation_support >= min_support and confidence >= min_confidence.
'''
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from declare.templates import TEMPLATES, DEFAULT_TEMPLATES, PAD, activation_mask
from declare.selection import prune_redundant, sort_constraints


@dataclass
class DeclareConstraint:
    template: str
    a: int
    b: Optional[int] = None
    n: int = 1
    a_name: Optional[str] = None
    b_name: Optional[str] = None
    support: float = 0.0
    activation_support: float = 0.0
    confidence: float = 0.0
    confidence_pos: Optional[float] = None
    confidence_neg: Optional[float] = None
    # Set by PM4PyDeclareMiner: the pm4py template and the stats under the LTN predicate semantics
    pm4py_template: Optional[str] = None
    ltn_support: Optional[float] = None
    ltn_confidence: Optional[float] = None

    @property
    def spec(self):
        return TEMPLATES[self.template]

    @property
    def key(self):
        return (self.template, self.a, self.b, self.n)

    @property
    def name(self):
        '''Readable identifier, e.g. "Response(ER Triage, ER Sepsis Triage)".'''
        a = self.a_name or str(self.a)
        if self.spec.arity == 1:
            args = f"{a}, {self.n}" if self.template in ("existence", "absence", "exactly") else a
        else:
            args = f"{a}, {self.b_name or self.b}"
        return f"{self.spec.display_name}({args})"

    def __str__(self):
        name, args = self.name.split("(", 1)
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
        stats = f"supp={self.support:.3f} act={self.activation_support:.3f} conf={self.confidence:.3f}"
        if self.confidence_pos is not None or self.confidence_neg is not None:
            stats += f" conf+={fmt(self.confidence_pos)} conf-={fmt(self.confidence_neg)}"
        if self.pm4py_template is not None:
            stats += f" ltn_conf={fmt(self.ltn_confidence)}"
            name = f"{name}<pm4py:{self.pm4py_template}>"
        return f"{name}({args}  [{stats}]"


def activity_vocab(activities):
    '''Activity name -> code, matching `pd.Categorical(col).cat.codes + 1` in the preprocessing.'''
    return {name: i + 1 for i, name in enumerate(pd.Categorical(activities).categories)}


def encode_log(data, vocab, max_len=None, case_col="case:concept:name", activity_col="concept:name",
               label_col="label"):
    '''
    Turn an event log DataFrame into a (num_cases, L) LongTensor of activity codes, right-padded
    with PAD. Events keep the DataFrame order within each case (as in create_ngrams).
    Returns traces, case ids and per-case labels (None if label_col is missing).
    '''
    data = data[[c for c in (case_col, activity_col, label_col) if c in data.columns]].copy()
    data["_pos"] = data.groupby(case_col, sort=False).cumcount()
    if max_len is not None:
        data = data[data["_pos"] < max_len]
    case_ids, case_idx = np.unique(data[case_col].astype(str).to_numpy(), return_inverse=True)
    seq_len = int(data["_pos"].max()) + 1
    traces = np.full((len(case_ids), seq_len), PAD, dtype=np.int64)
    traces[case_idx, data["_pos"].to_numpy()] = data[activity_col].map(vocab).to_numpy()
    labels = None
    if label_col in data.columns:
        labels = data.groupby(case_idx)[label_col].first().to_numpy()
        labels = torch.tensor(labels, dtype=torch.long)
    return torch.as_tensor(traces), list(case_ids), labels


def activity_slice(x, feature_names, seq_len, activity_feature="concept:name"):
    '''
    Extract the activity sequence from the flattened, feature-major model input
    x of shape (B, num_features * seq_len) built by create_ngrams.
    '''
    offset = feature_names.index(activity_feature) * seq_len
    return x[..., offset:offset + seq_len].round().long()


class DeclareMiner:
    def __init__(self, templates=None, min_support=0.05, min_confidence=0.9, min_activity_support=0.0,
                 prefix_safe_only=False, prune_redundant=True, max_constraints=None,
                 absence_n=(2,), device="cpu", max_chunk_elements=2 ** 26):
        '''
        templates: template names to mine (default: templates.DEFAULT_TEMPLATES)
        min_support: minimum activation support (fraction of traces activating the constraint)
        min_confidence: minimum fraction of activated traces that satisfy the constraint
        min_activity_support: activities occurring in fewer traces than this fraction are skipped
        prefix_safe_only: keep only templates whose violations on a prefix are permanent
        prune_redundant: drop constraints implied by a stronger mined one on the same activities
        absence_n: values of n mined for Absence(a, n) (n=1 means "a never occurs")
        '''
        templates = list(templates or DEFAULT_TEMPLATES)
        if prefix_safe_only:
            templates = [t for t in templates if TEMPLATES[t].prefix_safe]
        unknown = set(templates) - set(TEMPLATES)
        if unknown:
            raise ValueError(f"Unknown Declare templates: {sorted(unknown)}")
        self.templates = templates
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_activity_support = min_activity_support
        self.prune_redundant = prune_redundant
        self.max_constraints = max_constraints
        self.absence_n = tuple(absence_n)
        self.device = device
        self.max_chunk_elements = max_chunk_elements
        self.constraints: List[DeclareConstraint] = []

    def fit(self, traces, labels=None, vocab: Optional[Dict[str, int]] = None):
        '''
        traces: (N, L) integer tensor of activity codes (PAD = 0), e.g. from encode_log or activity_slice
        labels: optional (N,) binary tensor, used to report confidence on positive / negative traces
        vocab: optional activity name -> code mapping, used for readable constraint names
        '''
        traces = torch.as_tensor(traces).long().to(self.device)
        names = {code: name for name, code in (vocab or {}).items()}

        # Identical traces (very common among prefixes) are evaluated once and weighted
        variants, inverse = torch.unique(traces, dim=0, return_inverse=True)
        weights = {"all": torch.bincount(inverse, minlength=len(variants)).float()}
        if labels is not None:
            labels = torch.as_tensor(labels).to(self.device).round().long()
            weights["pos"] = torch.bincount(inverse, weights=(labels == 1).float(), minlength=len(variants))
            weights["neg"] = torch.bincount(inverse, weights=(labels == 0).float(), minlength=len(variants))
        total = weights["all"].sum()

        codes = torch.unique(variants)
        codes = codes[codes != PAD]
        masks = variants.unsqueeze(0) == codes.view(-1, 1, 1)  # (A, V, L)
        activity_support = (masks.any(dim=-1).float() @ weights["all"]) / total
        keep = activity_support >= self.min_activity_support
        codes, masks = codes[keep], masks[keep]
        valid = variants != PAD

        candidates = []
        for template in self.templates:
            spec = TEMPLATES[template]
            if spec.arity == 1:
                for n in (self.absence_n if template == "absence" else (1,)):
                    candidates += self._score(template, masks, None, valid, weights, total, n=n)
            else:
                ia, ib = torch.meshgrid(torch.arange(len(codes)), torch.arange(len(codes)), indexing="ij")
                pair_mask = ia < ib if spec.symmetric else ia != ib
                candidates += self._score(template, masks, (ia[pair_mask], ib[pair_mask]), valid, weights, total)

        constraints = []
        for template, ia, ib, n, supp, act_supp, conf, conf_pos, conf_neg in candidates:
            a = int(codes[ia])
            b = int(codes[ib]) if ib is not None else None
            constraints.append(DeclareConstraint(
                template=template, a=a, b=b, n=n, a_name=names.get(a), b_name=names.get(b),
                support=supp, activation_support=act_supp, confidence=conf,
                confidence_pos=conf_pos, confidence_neg=conf_neg))

        if self.prune_redundant:
            constraints = prune_redundant(constraints)
        constraints = sort_constraints(constraints)
        if self.max_constraints is not None:
            constraints = constraints[:self.max_constraints]
        self.constraints = constraints
        return constraints

    def _score(self, template, masks, pairs, valid, weights, total, n=1):
        '''Evaluate one template on all candidate activities (or pairs) and return the ones passing the thresholds.'''
        spec = TEMPLATES[template]
        num_variants, seq_len = valid.shape
        if pairs is None:
            ia_all, ib_all = torch.arange(len(masks)), None
        else:
            ia_all, ib_all = pairs
        chunk = max(1, self.max_chunk_elements // max(1, num_variants * seq_len))
        results = []
        for start in range(0, len(ia_all), chunk):
            ia = ia_all[start:start + chunk]
            mask_a = masks[ia]
            mask_b = masks[ib_all[start:start + chunk]] if ib_all is not None else torch.zeros_like(mask_a)
            if spec.arity == 1:
                sat = spec.fn(mask_a, valid, n)
            else:
                sat = spec.fn(mask_a, mask_b, valid)
            act = activation_mask(template, mask_a, mask_b)  # (P, V)
            stats = {}
            for key, w in weights.items():
                n_act = act.float() @ w
                n_sat_act = (sat & act).float() @ w
                stats[key] = (n_act, n_sat_act)
            n_act, n_sat_act = stats["all"]
            supp = (sat.float() @ weights["all"]) / total
            act_supp = n_act / total
            conf = n_sat_act / n_act.clamp(min=1e-9)
            ok = (act_supp >= self.min_support) & (conf >= self.min_confidence) & (n_act > 0)
            for p in ok.nonzero(as_tuple=True)[0].tolist():
                conf_pos = conf_neg = None
                if "pos" in stats:
                    conf_pos = _ratio(stats["pos"][1][p], stats["pos"][0][p])
                    conf_neg = _ratio(stats["neg"][1][p], stats["neg"][0][p])
                results.append((template, int(ia[p]), int(ib_all[start + p]) if ib_all is not None else None, n,
                                float(supp[p]), float(act_supp[p]), float(conf[p]), conf_pos, conf_neg))
        return results

    def save(self, path, vocab=None, **metadata):
        params = {"templates": self.templates, "min_support": self.min_support,
                  "min_confidence": self.min_confidence, "min_activity_support": self.min_activity_support,
                  "prune_redundant": self.prune_redundant, **metadata}
        save_constraints(path, self.constraints, vocab=vocab, params=params)


def _ratio(num, den):
    return float(num / den) if den > 0 else None


def save_constraints(path, constraints, vocab=None, params=None):
    payload = {"params": params or {}, "vocab": vocab, "constraints": [asdict(c) for c in constraints]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_constraints(path):
    with open(path) as f:
        payload = json.load(f)
    return [DeclareConstraint(**c) for c in payload["constraints"]], payload.get("vocab")
