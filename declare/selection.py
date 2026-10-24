'''
Ordering and pruning of discovered Declare constraints.

    constraints = sort_constraints(constraints, by=["confidence", "activation_support"])
    constraints = prune_constraints(constraints, min_confidence=0.95, max_ltn_gap=0.01, redundant=True,
                                    transitive=True, top_k_per_template=5, top_k=30)

Metrics usable in `by` (descending unless suffixed with ":asc"; ltn_gap is ascending by default):
    support, activation_support, confidence, confidence_pos, confidence_neg,
    ltn_support, ltn_confidence, score (= activation_support * confidence),
    class_gap (= |confidence_pos - confidence_neg|), ltn_gap (= |confidence - ltn_confidence|)
Constraints with a missing metric are always placed last.
'''
from collections import defaultdict

from declare.templates import TEMPLATES, implied_templates


def _gap(x, y):
    return None if x is None or y is None else abs(x - y)


METRICS = {
    "support": lambda c: c.support,
    "activation_support": lambda c: c.activation_support,
    "confidence": lambda c: c.confidence,
    "confidence_pos": lambda c: c.confidence_pos,
    "confidence_neg": lambda c: c.confidence_neg,
    "ltn_support": lambda c: c.support if c.pm4py_template is None else c.ltn_support,
    "ltn_confidence": lambda c: c.confidence if c.pm4py_template is None else c.ltn_confidence,
    "score": lambda c: c.activation_support * c.confidence,
    "class_gap": lambda c: _gap(c.confidence_pos, c.confidence_neg),
    # Constraints mined by the tensor miner already use the LTN semantics
    "ltn_gap": lambda c: 0.0 if c.pm4py_template is None else _gap(c.confidence, c.ltn_confidence),
}
ASCENDING_BY_DEFAULT = {"ltn_gap"}
DEFAULT_SORT = ("confidence", "activation_support")

# Templates T for which T(a, b) and T(b, c) imply T(a, c) under the LTN (standard Declare) semantics
TRANSITIVE_TEMPLATES = ("responded_existence", "response", "precedence", "succession")


def _parse_key(key):
    name, _, order = key.partition(":")
    if name not in METRICS:
        raise ValueError(f"Unknown metric '{name}', choose from {list(METRICS)}")
    if order not in ("", "asc", "desc"):
        raise ValueError(f"Unknown sort order '{order}' in '{key}', use ':asc' or ':desc'")
    descending = (order == "desc") if order else name not in ASCENDING_BY_DEFAULT
    return METRICS[name], descending


def sort_constraints(constraints, by=DEFAULT_SORT):
    '''Stable multi-key sort; ties are broken by template and activity codes.'''
    if isinstance(by, str):
        by = [by]
    out = sorted(constraints, key=lambda c: (c.template, c.a, c.b or 0, c.n))
    for metric, descending in reversed([_parse_key(k) for k in by]):
        def sort_key(c, metric=metric, descending=descending):
            v = metric(c)
            return (v is None, 0.0 if v is None else (-v if descending else v))
        out.sort(key=sort_key)
    return out


def prune_redundant(constraints, tolerance=None):
    '''
    Remove constraints logically implied by another constraint on the same activities
    (e.g. Response(a, b) when ChainResponse(a, b) is kept). With a tolerance, the weaker
    constraint is only removed if the stronger one has confidence >= weaker confidence - tolerance,
    so that a weak strong-template rule does not hide a much more reliable weaker rule.
    '''
    by_key = {(c.template, c.a, c.b): c for c in constraints}
    removed = set()
    for (template, a, b), strong in by_key.items():
        for weaker in implied_templates(template):
            pairs = [(a, b), (b, a)] if TEMPLATES[template].symmetric else [(a, b)]
            for pa, pb in pairs:
                weak = by_key.get((weaker, pa, pb))
                if weak is not None and (tolerance is None or strong.confidence >= weak.confidence - tolerance):
                    removed.add((weaker, pa, pb))
    return [c for c in constraints if (c.template, c.a, c.b) not in removed]


def prune_transitive(constraints, templates=TRANSITIVE_TEMPLATES):
    '''
    Transitive reduction: drop T(a, c) when T(a, b) and T(b, c) are kept, for transitive templates T.
    Edges are tried from the least to the most confident, and an edge is only removed while a
    two-step path still exists, so the transitive closure of the kept constraints never shrinks.
    '''
    removed = set()
    for template in templates:
        edges = {(c.a, c.b): c for c in constraints if c.template == template and c.b is not None}
        succ = defaultdict(set)
        for a, b in edges:
            succ[a].add(b)
        for (a, c), con in sorted(edges.items(), key=lambda kv: (kv[1].confidence, kv[1].activation_support)):
            if any(c in succ[b] for b in succ[a] if b != c):
                succ[a].discard(c)
                removed.add(id(con))
    return [c for c in constraints if id(c) not in removed]


def _matches_template(c, names):
    return c.template in names or (c.pm4py_template is not None and c.pm4py_template in names)


def _codes(c):
    return [x for x in (c.a, c.b) if x is not None]


def _names(c):
    return [x for x in (c.a_name, c.b_name) if x is not None]


def prune_constraints(constraints, *, min_support=None, min_activation_support=None, min_confidence=None,
                      min_ltn_confidence=None, max_ltn_gap=None, min_class_gap=None, prefix_safe_only=False,
                      templates=None, exclude_templates=None, activities=None, redundant=False,
                      redundancy_tolerance=None, transitive=False, sort_by=DEFAULT_SORT, top_k_per_template=None,
                      top_k=None):
    '''
    Filter, reduce and rank constraints. Steps, in order:
      1. thresholds: min_support, min_activation_support, min_confidence, min_ltn_confidence,
         max_ltn_gap (|conf - ltn_conf|), min_class_gap (|conf+ - conf-|, constraints that
         behave differently on positive and negative traces)
      2. selection: prefix_safe_only, templates / exclude_templates (LTN or pm4py names),
         activities (names or codes; every activity of a constraint must be in the set)
      3. redundant: drop constraints implied by a kept one on the same activities (see prune_redundant)
      4. transitive: transitive reduction of Response/Precedence/Succession/RespondedExistence
      5. sort by sort_by, keep the first top_k_per_template per template, then the first top_k
    '''
    thresholds = [
        ("support", min_support, False), ("activation_support", min_activation_support, False),
        ("confidence", min_confidence, False), ("ltn_confidence", min_ltn_confidence, False),
        ("ltn_gap", max_ltn_gap, True), ("class_gap", min_class_gap, False),
    ]
    out = list(constraints)
    for metric, value, is_max in thresholds:
        if value is not None:
            fn = METRICS[metric]
            out = [c for c in out if fn(c) is not None and (fn(c) <= value if is_max else fn(c) >= value)]

    if prefix_safe_only:
        out = [c for c in out if c.spec.prefix_safe]
    if templates is not None:
        out = [c for c in out if _matches_template(c, set(templates))]
    if exclude_templates is not None:
        out = [c for c in out if not _matches_template(c, set(exclude_templates))]
    if activities is not None:
        allowed = set(activities)
        out = [c for c in out if all(x in allowed for x in _codes(c)) or (
            len(_names(c)) == len(_codes(c)) and all(x in allowed for x in _names(c)))]

    if redundant:
        out = prune_redundant(out, tolerance=redundancy_tolerance)
    if transitive:
        out = prune_transitive(out)

    out = sort_constraints(out, sort_by)
    if top_k_per_template is not None:
        counts = defaultdict(int)
        kept = []
        for c in out:
            if counts[c.template] < top_k_per_template:
                counts[c.template] += 1
                kept.append(c)
        out = kept
    if top_k is not None:
        out = out[:top_k]
    return out
