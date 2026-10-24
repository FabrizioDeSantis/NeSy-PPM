'''
Declare (control-flow) constraints. They describe the process and are never connected to the label: no constraint
implies P(x). With --declare in_p their truth values on a prefix are extra inputs of P, i.e. P(x, C(x)); with
--declare axioms they are axioms Forall x: C(x) of the LTN knowledge base (they involve no learnable predicate, so
they do not change the gradients of P). Two sources (--declare_source):

mined (default): the best constraints discovered from the training prefixes
    1. Declare constraints are mined with pm4py on the training prefixes only (the inputs the models are trained
       on, so no test case is used and the activity codes are the model's ones)
    2. only high-quality constraints are kept: prefix-safe (once violated on a prefix, violated on every
       continuation, so the input feature is stable as the trace grows), confidence >= declare_min_confidence,
       same semantics in pm4py and in the LTN predicates (|conf - ltn_conf| <= declare_max_ltn_gap),
       no redundant / transitively implied constraints, no duplicates (same truth values on every training prefix)
    3. they are ranked by class gap |conf+ - conf-| (the constraints whose confidence differs most between label-1
       and label-0 prefixes, i.e. the most informative for the outcome), and the first declare_top_k are kept

predefined: the hand-written constraints of the knowledge base (PREDEFINED_CONSTRAINTS)
'''
import json
from dataclasses import asdict, dataclass

import torch

from declare import DeclareKB, PM4PyDeclareMiner, activity_slice, prune_constraints
from declare.miner import DeclareConstraint
from declare.templates import activation_mask, evaluate

DECLARE_MODES = ["none", "in_p", "axioms"]
DECLARE_SOURCES = ["mined", "predefined"]

# Knowledge base: (rule id, template, a, b, description)
PREDEFINED_CONSTRAINTS = {
    "sepsis": [
        ("R4", "chain_response", "ER Triage", "ER Sepsis Triage", "ER Sepsis Triage immediately follows ER Triage"),
        ("R5", "response", "ER Sepsis Triage", "IV Antibiotics", "IV Antibiotics occurs after ER Sepsis Triage"),
        ("R6", "response", "ER Sepsis Triage", "IV Liquid", "IV Liquid occurs after ER Sepsis Triage"),
    ],
    "bpi12": [
        ("R4", "response", "A_ACCEPTED-COMPLETE", "W_Completeren aanvraag-COMPLETE",
         "W_Completeren aanvraag occurs after A_ACCEPTED"),
        # R5, Response(W_Valideren aanvraag-COMPLETE, O_ACCEPTED-COMPLETE), cannot be grounded: O_ACCEPTED defines
        # the label and is not in the processed log
        ("R6", "response", "O_SENT_BACK-COMPLETE", "W_Valideren aanvraag-COMPLETE",
         "W_Valideren aanvraag occurs after O_SENT_BACK"),
    ],
    "bpi17": [
        ("R4", "response", "A_Submitted", "A_Accepted", "A_Accepted occurs after A_Submitted"),
        ("R5", "response", "A_Accepted", "O_Create Offer", "O_Create Offer occurs after A_Accepted"),
        ("R6", "response", "A_Complete", "W_Validate application", "W_Validate application occurs after A_Complete"),
    ],
    "traffic": [
        ("R4", "chain_response", "Create Fine", "Send Fine", "Send Fine immediately follows Create Fine"),
        ("R5", "chain_response", "Send Fine", "Insert Fine Notification",
         "Insert Fine Notification immediately follows Send Fine"),
        # The knowledge base names the target 'Payment Notification', which is not an activity of the log
        ("R6", "response", "Send Fine", "Payment", "Payment occurs after Send Fine"),
    ],
}


@dataclass
class DeclareKnowledge:
    source: str
    kb: DeclareKB

    @property
    def constraints(self):
        return self.kb.constraints

    @property
    def names(self):
        return [c.name for c in self.kb.constraints]

    def truth(self, x):
        '''(B, K) bool: satisfaction of each constraint on the prefixes in x.'''
        return self.kb.satisfaction(x) > 0.5

    def axioms(self, x_var):
        '''One axiom Forall x: C(x) per constraint.'''
        return self.kb.axioms(x_var, mode="satisfy")


def add_declare_args(parser):
    group = parser.add_argument_group("Declare constraints")
    group.add_argument("--declare", default="none", choices=DECLARE_MODES,
                       help="none: no Declare constraints (default); in_p: their truth values are extra inputs of P, "
                            "P(x, C(x)); axioms: axioms Forall x: C(x) that do not involve P")
    group.add_argument("--declare_source", default="mined", choices=DECLARE_SOURCES,
                       help="mined: best constraints mined from the training prefixes (default); "
                            "predefined: hand-written constraints of the knowledge base")
    group.add_argument("--declare_top_k", type=int, default=10, help="Number of mined constraints, ranked by class gap")
    group.add_argument("--declare_min_support", type=float, default=0.05,
                       help="Minimum activation support of the mined constraints")
    group.add_argument("--declare_min_confidence", type=float, default=0.9,
                       help="Minimum confidence of the mined constraints")
    group.add_argument("--declare_max_ltn_gap", type=float, default=0.01,
                       help="Maximum |pm4py confidence - LTN confidence| of the mined constraints")


def build_declare_knowledge(data, args, output_path=None):
    knowledge = predefined_declare_knowledge(data) if args.declare_source == "predefined" \
        else mine_declare_knowledge(data, args)
    print(f"   Declare constraints ({knowledge.source}):")
    for c in knowledge.constraints:
        print(f"   {c}")
    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump({"source": knowledge.source, "constraints": [asdict(c) for c in knowledge.constraints]},
                      f, indent=2)
        print(f"   saved to {output_path}")
    return knowledge


def activity_codes(data):
    '''Activity name -> code, as encoded by the preprocessing.'''
    return data.category_codes["concept:name"]


def mine_declare_knowledge(data, args):
    layout = data.layout
    print("-- Mining Declare constraints on the training prefixes")
    vocab = activity_codes(data)
    acts = activity_slice(data.train.sequences, list(layout.feature_names), layout.seq_len)
    miner = PM4PyDeclareMiner(min_support=args.declare_min_support, min_confidence=args.declare_min_confidence,
                              prefix_safe_only=True, prune_redundant=False)
    mined = miner.fit(acts, data.train.labels, vocab=vocab)
    pruned = prune_constraints(mined, max_ltn_gap=args.declare_max_ltn_gap, redundant=True, transitive=True,
                               sort_by=["class_gap", "confidence", "activation_support"])
    kb = DeclareKB(pruned, list(layout.feature_names), layout.seq_len, vocab=vocab)
    constraints = drop_duplicates(kb, data.train.sequences)[:args.declare_top_k]
    print(f"   {len(mined)} prefix-safe constraints mined, {len(pruned)} after pruning, best {len(constraints)} kept")
    return DeclareKnowledge("mined", kb._with_constraints(constraints))


def drop_duplicates(kb, x):
    '''Constraints of kb (in order) without those with the same truth values as a previous one on every prefix of x.'''
    satisfied = kb.satisfaction(x) > 0.5
    kept, seen = [], set()
    for k, c in enumerate(kb.constraints):
        signature = satisfied[:, k].numpy().tobytes()
        if signature not in seen:
            seen.add(signature)
            kept.append(c)
    return kept


def predefined_declare_knowledge(data):
    print("-- Predefined Declare constraints (knowledge base)")
    vocab = activity_codes(data)
    acts = activity_slice(data.train.sequences, list(data.layout.feature_names), data.layout.seq_len)
    y = data.train.labels.round().long()
    constraints = [constraint_with_stats(template, vocab[a], vocab[b], a, b, acts, y)
                   for _, template, a, b, _ in PREDEFINED_CONSTRAINTS[data.name]]
    return DeclareKnowledge("predefined", DeclareKB(constraints, list(data.layout.feature_names), data.layout.seq_len,
                                                    vocab=vocab))


def constraint_with_stats(template, a, b, a_name, b_name, acts, y):
    '''DeclareConstraint with the statistics computed by the miner (support, confidence, conf+ / conf-) on acts.'''
    stats = constraint_stats(template, a, b, acts, y)
    return DeclareConstraint(template, a, b, a_name=a_name, b_name=b_name, support=stats["support"],
                             activation_support=stats["activated"], confidence=stats["confidence"] or 0.0,
                             confidence_pos=stats["confidence_pos"], confidence_neg=stats["confidence_neg"])


def constraint_stats(template, a, b, acts, y):
    '''
    Statistics of one constraint on traces (or prefixes) of activity codes acts (N, L) with labels y:
        support:     fraction of the traces satisfying it (vacuously included)
        activated:   fraction of the traces activating it
        confidence:  fraction of the activated traces satisfying it; confidence_pos / _neg: the same on the
                     traces with label 1 / label 0
    '''
    satisfied = evaluate(template, acts, a, b if b is not None else -1)
    activated = activation_mask(template, acts == a, acts == (b if b is not None else -1))

    def confidence(mask):
        n = int((activated & mask).sum())
        return int((activated & satisfied & mask).sum()) / n if n else None

    everything = torch.ones_like(satisfied)
    return {
        "support": float(satisfied.float().mean()) if len(acts) else None,
        "activated": float(activated.float().mean()) if len(acts) else None,
        "confidence": confidence(everything),
        "confidence_pos": confidence(y == 1),
        "confidence_neg": confidence(y == 0),
    }
