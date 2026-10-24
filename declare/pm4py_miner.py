'''
Declare discovery with pm4py (Maggi et al., CIDM 2011, as implemented in pm4py.discover_declare).

pm4py evaluates every candidate rule on every trace as +1 (activated and satisfied),
-1 (activated and violated) or 0 (not activated), and reports for each rule
    "support"    = number of activated traces
    "confidence" = number of activated traces satisfying the rule.
Here the same per-trace evaluation (pm4py's form_rules_table) is run once per trace variant and
the counts are weighted by the variant frequencies, which gives exactly the counts of
pm4py.discover_declare on the full log, and also allows splitting them by label. They are turned
into the same ratios reported by DeclareMiner:
    support            = fraction of traces not violating the rule (vacuous satisfaction included)
    activation_support = pm4py support / number of traces
    confidence         = pm4py confidence / pm4py support
    confidence_pos/neg = confidence on positive / negative traces

pm4py's semantics differ from standard Declare for some templates (e.g. its Precedence family is
not activated when the target occurs without the source, its AltResponse/ChainResponse require the
same number of occurrences of both activities). Since the LTN predicates use the standard semantics
of declare.templates, every mined constraint is also re-evaluated with them (ltn_support,
ltn_confidence): constraints where the two disagree are not enforced as pm4py measured them.
'''
import torch
from pm4py.algo.discovery.declare import templates as pm4py_templates
from pm4py.algo.discovery.declare.variants import classic as pm4py_declare
from pm4py.objects.log.obj import Event, EventLog, Trace

from declare.miner import DeclareConstraint, DeclareMiner
from declare.selection import prune_redundant, sort_constraints
from declare.templates import PAD, TEMPLATES, activation_mask

# pm4py template -> (LTN template, n). pm4py's ExactlyOne is only activated when the activity occurs,
# so as a constraint it reads "at most once", i.e. Absence(a, 2)
PM4PY_TO_LTN = {
    pm4py_templates.EXISTENCE: ("existence", 1),
    pm4py_templates.EXACTLY_ONE: ("absence", 2),
    pm4py_templates.INIT: ("init", 1),
    pm4py_templates.ABSENCE: ("absence", 1),
    pm4py_templates.RESPONDED_EXISTENCE: ("responded_existence", 1),
    pm4py_templates.COEXISTENCE: ("co_existence", 1),
    pm4py_templates.RESPONSE: ("response", 1),
    pm4py_templates.PRECEDENCE: ("precedence", 1),
    pm4py_templates.SUCCESSION: ("succession", 1),
    pm4py_templates.ALTRESPONSE: ("alternate_response", 1),
    pm4py_templates.ALTPRECEDENCE: ("alternate_precedence", 1),
    pm4py_templates.ALTSUCCESSION: ("alternate_succession", 1),
    pm4py_templates.CHAINRESPONSE: ("chain_response", 1),
    pm4py_templates.CHAINPRECEDENCE: ("chain_precedence", 1),
    pm4py_templates.CHAINSUCCESSION: ("chain_succession", 1),
    pm4py_templates.NONCOEXISTENCE: ("not_co_existence", 1),
    pm4py_templates.NONSUCCESSION: ("not_succession", 1),
    pm4py_templates.NONCHAINSUCCESSION: ("not_chain_succession", 1),
}
PM4PY_TEMPLATES = list(PM4PY_TO_LTN)

# pm4py only computes derived templates if the templates they are built from are allowed
_PM4PY_DEPENDENCIES = {
    pm4py_templates.ABSENCE: {pm4py_templates.EXISTENCE},
    pm4py_templates.SUCCESSION: {pm4py_templates.RESPONSE, pm4py_templates.PRECEDENCE},
    pm4py_templates.ALTSUCCESSION: {pm4py_templates.ALTRESPONSE, pm4py_templates.ALTPRECEDENCE},
    pm4py_templates.CHAINSUCCESSION: {pm4py_templates.CHAINRESPONSE, pm4py_templates.CHAINPRECEDENCE},
    pm4py_templates.COEXISTENCE: {pm4py_templates.RESPONDED_EXISTENCE},
    pm4py_templates.NONCOEXISTENCE: {pm4py_templates.COEXISTENCE, pm4py_templates.RESPONDED_EXISTENCE},
    pm4py_templates.NONSUCCESSION: {pm4py_templates.SUCCESSION, pm4py_templates.RESPONSE,
                                    pm4py_templates.PRECEDENCE},
    pm4py_templates.NONCHAINSUCCESSION: {pm4py_templates.CHAINSUCCESSION, pm4py_templates.CHAINRESPONSE,
                                         pm4py_templates.CHAINPRECEDENCE},
}


def _with_dependencies(templates):
    allowed, stack = set(), list(templates)
    while stack:
        t = stack.pop()
        if t not in allowed:
            allowed.add(t)
            stack.extend(_PM4PY_DEPENDENCIES.get(t, ()))
    return allowed


def variants_to_event_log(variants, names):
    '''One pm4py trace per encoded variant (PAD removed), in the same order.'''
    log = EventLog()
    for i, row in enumerate(variants.tolist()):
        trace = Trace([Event({"concept:name": names[code]}) for code in row if code != PAD])
        trace.attributes["concept:name"] = str(i)
        log.append(trace)
    return log


class PM4PyDeclareMiner(DeclareMiner):
    def __init__(self, templates=None, min_support=0.05, min_confidence=0.9, min_activity_support=0.0,
                 prefix_safe_only=False, prune_redundant=True, max_constraints=None):
        '''
        templates: pm4py template names to mine (default: all of PM4PY_TEMPLATES)
        min_support: minimum activation support (pm4py support / number of traces)
        min_confidence: minimum pm4py confidence / pm4py support
        min_activity_support: drop rules on activities occurring in fewer traces than this fraction
            (activities are not removed from the traces, which would change the Chain* semantics)
        prefix_safe_only: keep only rules mapped to LTN templates that are prefix-safe
        prune_redundant: drop rules implied by a stronger mined rule on the same activities
        '''
        templates = list(templates or PM4PY_TEMPLATES)
        unknown = set(templates) - set(PM4PY_TEMPLATES)
        if unknown:
            raise ValueError(f"Unknown pm4py Declare templates: {sorted(unknown)}")
        if prefix_safe_only:
            templates = [t for t in templates if TEMPLATES[PM4PY_TO_LTN[t][0]].prefix_safe]
        self.templates = templates
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_activity_support = min_activity_support
        self.prune_redundant = prune_redundant
        self.max_constraints = max_constraints
        self.constraints = []

    def fit(self, traces, labels=None, vocab=None):
        '''Same inputs as DeclareMiner.fit: (N, L) activity codes, optional (N,) labels and name -> code vocab.'''
        traces = torch.as_tensor(traces).long().cpu()
        variants, inverse = torch.unique(traces, dim=0, return_inverse=True)
        weights = {"all": torch.bincount(inverse, minlength=len(variants)).double()}
        if labels is not None:
            labels = torch.as_tensor(labels).cpu().round().long()
            weights["pos"] = torch.bincount(inverse, weights=(labels == 1).double(), minlength=len(variants))
            weights["neg"] = torch.bincount(inverse, weights=(labels == 0).double(), minlength=len(variants))
        non_empty = (variants != PAD).any(dim=1)
        variants = variants[non_empty]
        weights = {k: w[non_empty] for k, w in weights.items()}
        total = weights["all"].sum().item()

        names = {code: name for name, code in (vocab or {}).items()}
        codes = [c for c in torch.unique(variants).tolist() if c != PAD]
        names = {c: names.get(c, str(c)) for c in codes}
        name_to_code = {name: c for c, name in names.items()}
        activity_support = {c: ((variants == c).any(dim=1).double() @ weights["all"]).item() / total for c in codes}

        # pm4py per-variant rule table: one row per variant, one column per rule, values in {-1, 0, 1}
        rules_df = pm4py_declare.form_rules_table(
            variants_to_event_log(variants, names),
            parameters={pm4py_declare.Parameters.ALLOWED_TEMPLATES: _with_dependencies(self.templates)})
        assert len(rules_df) == len(variants)

        constraints = {}
        for column in rules_df.columns:
            pm4py_template, acts = column[0], [x for x in column[1:] if isinstance(x, str) and x]
            if pm4py_template not in self.templates:
                continue
            template, n = PM4PY_TO_LTN[pm4py_template]
            a = name_to_code[acts[0]]
            b = name_to_code[acts[1]] if len(acts) > 1 else None
            if TEMPLATES[template].symmetric and a > b:
                continue
            if min(activity_support[c] for c in (a, b) if c is not None) < self.min_activity_support:
                continue
            values = torch.tensor(rules_df[column].to_numpy(), dtype=torch.long)
            counts = {k: (((values != 0).double() @ w).item(), ((values == 1).double() @ w).item())
                      for k, w in weights.items()}
            n_act, n_sat = counts["all"]
            if n_act == 0 or n_act / total < self.min_support or n_sat / n_act < self.min_confidence:
                continue
            constraint = DeclareConstraint(
                template=template, a=a, b=b, n=n, a_name=names[a], b_name=names.get(b),
                support=(total - n_act + n_sat) / total, activation_support=n_act / total,
                confidence=n_sat / n_act, pm4py_template=pm4py_template)
            if "pos" in counts:
                constraint.confidence_pos = _ratio(*counts["pos"][::-1])
                constraint.confidence_neg = _ratio(*counts["neg"][::-1])
            constraints[constraint.key] = constraint

        constraints = list(constraints.values())
        _add_ltn_stats(constraints, variants, weights["all"])
        if self.prune_redundant:
            constraints = prune_redundant(constraints)
        constraints = sort_constraints(constraints)
        if self.max_constraints is not None:
            constraints = constraints[:self.max_constraints]
        self.constraints = constraints
        return constraints


def _ratio(num, den):
    return num / den if den > 0 else None


def _add_ltn_stats(constraints, variants, weights):
    '''Support and confidence of each constraint under the semantics of the LTN predicates.'''
    total = weights.sum().item()
    valid = variants != PAD
    for c in constraints:
        mask_a = variants == c.a
        mask_b = variants == c.b if c.b is not None else torch.zeros_like(mask_a)
        if c.spec.arity == 1:
            sat = c.spec.fn(mask_a, valid, c.n)
        else:
            sat = c.spec.fn(mask_a, mask_b, valid)
        act = activation_mask(c.template, mask_a, mask_b)
        n_act = (act.double() @ weights).item()
        c.ltn_support = (sat.double() @ weights).item() / total
        c.ltn_confidence = _ratio((sat & act).double() @ weights, n_act)
        if c.ltn_confidence is not None:
            c.ltn_confidence = float(c.ltn_confidence)
