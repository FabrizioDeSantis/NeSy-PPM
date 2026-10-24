'''
Declare templates evaluated on batches of integer-encoded traces.

A trace is a row of activity codes, left-aligned and right-padded with PAD (codes start at 1,
as produced by `pd.Categorical(...).cat.codes + 1` in the preprocessing scripts).
Every template works on boolean occurrence masks of shape (..., L) and returns a (...,) bool
tensor, so the same code is used both by the miner (batched over constraints x traces) and by
the LTN predicates (batched over traces). Semantics are LTLf on the finite trace (or prefix).

Adapted from declare_ltn_templates.py (now legacy/declare_ltn_templates.py).
'''
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch

PAD = 0


def _positions(mask):
    return torch.arange(mask.shape[-1], device=mask.device).expand(mask.shape)


def _first_index(mask):
    # Index of the first True, L if none
    return torch.where(mask, _positions(mask), mask.shape[-1]).min(dim=-1).values


def _last_index(mask):
    # Index of the last True, -1 if none
    return torch.where(mask, _positions(mask), -1).max(dim=-1).values


def _next_index(mask):
    # For each position i, index of the first True strictly after i (L if none)
    seq_len = mask.shape[-1]
    idx = torch.where(mask, _positions(mask), seq_len)
    suffix_min = idx.flip(-1).cummin(dim=-1).values.flip(-1)
    nxt = torch.full_like(suffix_min, seq_len)
    nxt[..., :-1] = suffix_min[..., 1:]
    return nxt


def _prev_index(mask):
    # For each position i, index of the last True strictly before i (-1 if none)
    idx = torch.where(mask, _positions(mask), -1)
    prefix_max = idx.cummax(dim=-1).values
    prv = torch.full_like(prefix_max, -1)
    prv[..., 1:] = prefix_max[..., :-1]
    return prv


def _shift_left(mask):
    # out[i] = mask[i + 1]
    out = torch.zeros_like(mask)
    out[..., :-1] = mask[..., 1:]
    return out


def _shift_right(mask):
    # out[i] = mask[i - 1]
    out = torch.zeros_like(mask)
    out[..., 1:] = mask[..., :-1]
    return out


### UNARY TEMPLATES: f(a, valid, n)

def existence(a, valid, n=1):
    return a.sum(dim=-1) >= n

def absence(a, valid, n=1):
    # Absence(a, n): a occurs less than n times (Absence(a, 1) = a never occurs)
    return a.sum(dim=-1) < n

def exactly(a, valid, n=1):
    return a.sum(dim=-1) == n

def init(a, valid, n=1):
    return a[..., 0]

def end(a, valid, n=1):
    last = _last_index(valid)
    return torch.gather(a, -1, last.clamp(min=0).unsqueeze(-1)).squeeze(-1) & (last >= 0)


### BINARY TEMPLATES: f(a, b, valid)

def responded_existence(a, b, valid):
    return ~a.any(dim=-1) | b.any(dim=-1)

def co_existence(a, b, valid):
    return a.any(dim=-1) == b.any(dim=-1)

def response(a, b, valid):
    # Every a is eventually followed by b
    return ~a.any(dim=-1) | (_last_index(b) > _last_index(a))

def precedence(a, b, valid):
    # b occurs only if a occurred before
    return ~b.any(dim=-1) | (_first_index(a) < _first_index(b))

def succession(a, b, valid):
    return response(a, b, valid) & precedence(a, b, valid)

def alternate_response(a, b, valid):
    # Every a is followed by b before the next a
    return (~a | (_next_index(b) < _next_index(a))).all(dim=-1)

def alternate_precedence(a, b, valid):
    # Every b is preceded by a after the previous b
    return (~b | (_prev_index(a) > _prev_index(b))).all(dim=-1)

def alternate_succession(a, b, valid):
    return alternate_response(a, b, valid) & alternate_precedence(a, b, valid)

def chain_response(a, b, valid):
    # Every a is immediately followed by b
    return (~a | _shift_left(b)).all(dim=-1)

def chain_precedence(a, b, valid):
    # Every b is immediately preceded by a
    return (~b | _shift_right(a)).all(dim=-1)

def chain_succession(a, b, valid):
    return chain_response(a, b, valid) & chain_precedence(a, b, valid)

def not_co_existence(a, b, valid):
    return ~(a.any(dim=-1) & b.any(dim=-1))

def not_succession(a, b, valid):
    # No b after any a
    return ~(_first_index(a) < _last_index(b))

def not_chain_succession(a, b, valid):
    # a is never immediately followed by b
    return ~(a & _shift_left(b)).any(dim=-1)


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    fn: Callable
    arity: int
    # Which activity activates the constraint: "a", "b", "either" or None (always active)
    activation: Optional[str]
    # True if a violation on a prefix can't be fixed by any continuation, i.e. the
    # constraint can be soundly enforced on the prefixes the models are trained on
    prefix_safe: bool
    # Templates logically implied by this one on the same (a, b), used to prune redundancy
    implies: Tuple[str, ...] = field(default_factory=tuple)
    # f(a, b) == f(b, a): only one ordering of each pair is mined
    symmetric: bool = False

    @property
    def display_name(self):
        return "".join(part.capitalize() for part in self.name.split("_"))


TEMPLATES = {spec.name: spec for spec in [
    TemplateSpec("existence", existence, 1, None, False),
    TemplateSpec("absence", absence, 1, None, True),
    TemplateSpec("exactly", exactly, 1, None, False),
    TemplateSpec("init", init, 1, None, True),
    TemplateSpec("end", end, 1, None, False),
    TemplateSpec("responded_existence", responded_existence, 2, "a", False),
    TemplateSpec("co_existence", co_existence, 2, "either", False, ("responded_existence",), symmetric=True),
    TemplateSpec("response", response, 2, "a", False, ("responded_existence",)),
    TemplateSpec("precedence", precedence, 2, "b", True),
    TemplateSpec("succession", succession, 2, "either", False, ("response", "precedence")),
    TemplateSpec("alternate_response", alternate_response, 2, "a", False, ("response",)),
    TemplateSpec("alternate_precedence", alternate_precedence, 2, "b", True, ("precedence",)),
    TemplateSpec("alternate_succession", alternate_succession, 2, "either", False,
                 ("alternate_response", "alternate_precedence", "succession")),
    TemplateSpec("chain_response", chain_response, 2, "a", False, ("alternate_response",)),
    TemplateSpec("chain_precedence", chain_precedence, 2, "b", True, ("alternate_precedence",)),
    TemplateSpec("chain_succession", chain_succession, 2, "either", False,
                 ("chain_response", "chain_precedence", "alternate_succession")),
    TemplateSpec("not_co_existence", not_co_existence, 2, "either", True, ("not_succession",), symmetric=True),
    TemplateSpec("not_succession", not_succession, 2, "a", True, ("not_chain_succession",)),
    TemplateSpec("not_chain_succession", not_chain_succession, 2, "a", True),
]}

# End is meaningless on prefixes and Exactly/Absence(n>1) explode the search space,
# so they are only mined when explicitly requested
DEFAULT_TEMPLATES = [name for name in TEMPLATES if name not in ("end", "exactly")]


def implied_templates(name):
    # Transitive closure of TemplateSpec.implies
    seen, stack = set(), list(TEMPLATES[name].implies)
    while stack:
        t = stack.pop()
        if t not in seen:
            seen.add(t)
            stack.extend(TEMPLATES[t].implies)
    return seen


def evaluate(template, traces, a, b=None, n=1):
    '''
    Evaluate a template on traces of shape (..., L) with activity codes a (and b).
    a and b are ints or tensors broadcastable to traces[..., :1]. Returns a (...,) bool tensor.
    '''
    spec = TEMPLATES[template]
    valid = traces != PAD
    mask_a = traces == a
    if spec.arity == 1:
        return spec.fn(mask_a, valid, n)
    return spec.fn(mask_a, traces == b, valid)


def activation_mask(template, mask_a, mask_b):
    '''Per-trace bool: whether the constraint is activated (non-vacuous).'''
    activation = TEMPLATES[template].activation
    if activation == "a":
        return mask_a.any(dim=-1)
    if activation == "b":
        return mask_b.any(dim=-1)
    if activation == "either":
        return mask_a.any(dim=-1) | mask_b.any(dim=-1)
    return torch.ones(mask_a.shape[:-1], dtype=torch.bool, device=mask_a.device)
