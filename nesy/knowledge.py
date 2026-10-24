'''
Domain knowledge of each dataset, as crisp rules on the model input.

A rule says: if condition(x) holds for a prefix x, its label should be `label`. Each rule is defined once here, on
named features. There are two kinds:
    R1-R3 (split=True)   the rules of NeSyPM-Compliance
    E1, ... (split=False) the additional rules of the original knowledge base (former main_<dataset>.py scripts)
and they are used as follows:
    LTN experiments   every rule: axiom  Forall x: condition(x) -> P(x)  (-> Not(P(x)) when label = 0), also in the
                      rule pruning (gating scores) and the adaptive rule weights
    rule_k columns    only R1-R3: rule Rk evaluated on the entire trace of each case (make_rule_labels.py), used by
                      the compliance split of data/preprocess_<dataset>.py
    evaluation        compliance on R1-R3: among the test prefixes where a rule fires and the true label agrees with
                      it, the fraction whose prediction agrees too; reported per rule for every rule
'''
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Rule:
    name: str
    label: int
    condition: Callable  # (B, width) model input -> (B,) bool
    formula: str = ""    # first-order formula, as in knowledge_base.txt
    split: bool = True   # R1-R3: a rule_k column, used by the compliance split and the compliance metric


class Knowledge:
    def __init__(self, rules):
        self.rules = list(rules)
        self.split_rules = [r for r in self.rules if r.split]

    def __len__(self):
        return len(self.rules)

    def truth(self, x, rules=None):
        '''(B, K) bool: truth of the antecedent (condition) of each rule (default: every rule) on x.'''
        rules = self.rules if rules is None else rules
        if not rules:
            return torch.zeros(len(x), 0, dtype=torch.bool, device=x.device)
        return torch.stack([r.condition(x) for r in rules], dim=1)

    def labels(self, rules=None, device=None):
        rules = self.rules if rules is None else rules
        return torch.tensor([r.label for r in rules], device=device)

    def print_summary(self, x, y):
        '''How often each rule fires on (x, y) and how often the label agrees with it.'''
        print("-- Rules (on the training prefixes)")
        truth = self.truth(x)
        for k, rule in enumerate(self.rules):
            fired = truth[:, k]
            n = int(fired.sum())
            agree = f"{int((y[fired] == rule.label).sum()) / n:.3f}" if n else "n/a"
            extra = "" if rule.split else "  [not in the compliance split]"
            print(f"   {rule.name}: fires on {n} prefixes, label agrees on {agree}{extra}")


### HELPERS

def scaled(scaler, value):
    '''`value` in the units of a feature normalized by a fitted MinMaxScaler.'''
    return float(value * scaler.scale_[0] + scaler.min_[0])


def unscaled(scaler, values):
    '''Inverse of the MinMaxScaler normalization (tensor of normalized values -> original units).'''
    return (values - float(scaler.min_[0])) / float(scaler.scale_[0])


def occurs(acts, a):
    '''(B,) activity a occurs in the prefix.'''
    return (acts == a).any(dim=1)


def occurs_before(acts, a, b):
    '''(B,) some occurrence of activity a precedes some occurrence of activity b.'''
    is_a = (acts == a).long()
    a_seen_before = (is_a.cumsum(dim=1) - is_a) > 0
    return (a_seen_before & (acts == b)).any(dim=1)


### RULES OF EACH DATASET
# Case attributes (e.g. the requested amount) have the same value at every event: they are read at the first one.

def bpi12_rules(data):
    layout, amount = data.layout, data.scalers["case:AMOUNT_REQ"]
    resource = data.category_codes["org:resource"]
    res_11169, res_10910 = resource["11169"], resource["10910"]
    activity = data.category_codes["concept:name"]
    cancelled, sent_back = activity["O_CANCELLED-COMPLETE"], activity["O_SENT_BACK-COMPLETE"]
    lt_10k, gt_40k, le_60k = scaled(amount, 10_000), scaled(amount, 40_000), scaled(amount, 60_000)

    def requested_amount(x):
        return layout.get(x, "case:AMOUNT_REQ")[:, 0]

    def resources(x):
        return layout.get(x, "org:resource")

    # Label 1: the application is accepted
    return [
        Rule("R1: amount < 10k -> not accepted", 0, lambda x: requested_amount(x) < lt_10k,
             formula="Forall x (RequestedAmount(x) < 10000 -> Not(Accepted(x)))"),
        Rule("R2: 40k < amount <= 60k -> not accepted", 0,
             lambda x: (requested_amount(x) > gt_40k) & (requested_amount(x) <= le_60k),
             formula="Forall x (RequestedAmount(x) > 40000 AND RequestedAmount(x) <= 60000 -> Not(Accepted(x)))"),
        Rule("R3: resource 11169 or 10910 involved -> not accepted", 0,
             lambda x: occurs(resources(x), res_11169) | occurs(resources(x), res_10910),
             formula="Forall x (HasResource(x, 10910) OR HasResource(x, 11169) -> Not(Accepted(x)))"),
        Rule("E1: O_CANCELLED at least 3 times -> not accepted", 0,
             lambda x: (layout.get(x, "concept:name") == cancelled).sum(dim=1) >= 3, split=False,
             formula="Forall x (Count(x, O_CANCELLED-COMPLETE) >= 3 -> Not(Accepted(x)))"),
        Rule("E2: O_SENT_BACK occurs -> not accepted", 0,
             lambda x: occurs(layout.get(x, "concept:name"), sent_back), split=False,
             formula="Forall x (HasActivity(x, O_SENT_BACK-COMPLETE) -> Not(Accepted(x)))"),
    ]


def bpi17_rules(data):
    layout, amount = data.layout, data.scalers["case:RequestedAmount"]
    create_offer = data.category_codes["concept:name"]["O_Create Offer"]
    loan_takeover = data.category_codes["case:LoanGoal"]["Existing loan takeover"]
    new_credit = data.category_codes["case:ApplicationType"]["New credit"]
    limit_20k = scaled(amount, 20_000)

    def requested_amount(x):
        return layout.get(x, "case:RequestedAmount")[:, 0]

    def offer_created(x):
        return occurs(layout.get(x, "concept:name"), create_offer)

    def has_credit_score(x):
        return (layout.get(x, "CreditScore") != 0).any(dim=1)

    # Label 1: the offer is accepted
    return [
        Rule("R1: amount < 20k, offer created with credit score -> accepted", 1,
             lambda x: (requested_amount(x) < limit_20k) & offer_created(x) & has_credit_score(x),
             formula="Forall x (RequestedAmount(x) < 20000 AND HasActivity(x, O_Create Offer) AND CreditScore(x) > 0 "
                     "-> Accepted(x))"),
        Rule("R2: offer created without credit score -> not accepted", 0,
             lambda x: offer_created(x) & ~has_credit_score(x),
             formula="Forall x (HasActivity(x, O_Create Offer) AND Not(CreditScore(x) > 0) -> Not(Accepted(x)))"),
        Rule("R3: amount > 20k, existing loan takeover -> not accepted", 0,
             lambda x: (requested_amount(x) > limit_20k) & occurs(layout.get(x, "case:LoanGoal"), loan_takeover),
             formula="Forall x (RequestedAmount(x) > 20000 AND LoanGoal(x) = \"Existing loan takeover\" "
                     "-> Not(Accepted(x)))"),
        Rule("E1: application type new credit -> accepted", 1,
             lambda x: occurs(layout.get(x, "case:ApplicationType"), new_credit), split=False,
             formula="Forall x (ApplicationType(x) = \"New credit\" -> Accepted(x))"),
    ]


def sepsis_rules(data):
    layout, scalers = data.layout, data.scalers
    activity = data.category_codes["concept:name"]
    crp, antibiotics = activity["CRP"], activity["IV Antibiotics"]
    lactic_2, crp_100 = scaled(scalers["LacticAcid"], 2), scaled(scalers["CRP"], 100)
    crp_10 = scaled(scalers["CRP"], 10)

    def flag(x, name):
        return layout.get(x, name).eq(1).any(dim=1)

    # Label 1: the patient is admitted to the ICU
    return [
        Rule("R1: lactic acid > 2 -> ICU", 1, lambda x: (layout.get(x, "LacticAcid") > lactic_2).any(dim=1),
             formula="Forall x (LacticAcid(x) > 2 -> ICU(x))"),
        Rule("R2: infection suspected, SIRS tachypnea and heart rate -> ICU", 1,
             lambda x: flag(x, "InfectionSuspected") & flag(x, "SIRSCritTachypnea") & flag(x, "SIRSCritHeartRate"),
             formula="Forall x (SuspectedInfection(x) AND Tachypnea(x) AND CriticalHeartRate(x) -> ICU(x))"),
        Rule("R3: CRP before IV antibiotics, CRP > 100 -> ICU", 1,
             lambda x: (occurs_before(layout.get(x, "concept:name"), crp, antibiotics)
                        & (layout.get(x, "CRP") > crp_100).any(dim=1)),
             formula="Forall x (Precedes(x, CRP, IV Antibiotics) AND CRPValue(x) > 100 -> ICU(x))"),
        # Only the CRP measurements count: CRP is 0 on padding and on the events of the other activities
        Rule("E1: CRP measured below 10 -> ICU", 1,
             lambda x: ((layout.get(x, "concept:name") == crp) & (layout.get(x, "CRP") < crp_10)).any(dim=1),
             split=False, formula="Forall x (CRPValue(x) < 10 -> ICU(x))"),
        Rule("E2: SIRS criteria 2 or more -> ICU", 1, lambda x: flag(x, "SIRSCriteria2OrMore"), split=False,
             formula="Forall x (SIRSCriteria2OrMore(x) -> ICU(x))"),
    ]


def traffic_rules(data):
    layout, scalers = data.layout, data.scalers
    activity = data.category_codes["concept:name"]
    add_penalty, payment = activity["Add penalty"], activity["Payment"]
    amount_400 = scaled(scalers["amount"], 400)
    vehicle_a = data.category_codes["vehicleClass"]["A"]

    def acts(x):
        return layout.get(x, "concept:name")

    def paid_less_than_fine(x):
        fine = unscaled(scalers["amount"], layout.get(x, "amount")[:, 0])
        paid = unscaled(scalers["paymentAmount"], layout.get(x, "paymentAmount")).max(dim=1).values
        return paid < fine

    # Label 1: the fine is sent for credit collection
    return [
        Rule("R1: paid less than the fine -> credit collection", 1,
             lambda x: occurs(acts(x), payment) & paid_less_than_fine(x),
             formula="Forall x (HasActivity(x, Payment) AND PaymentAmount(x) < FineAmount(x) -> CreditCollection(x))"),
        Rule("R2: penalty added -> credit collection", 1, lambda x: occurs(acts(x), add_penalty),
             formula="Forall x (HasActivity(x, Add penalty) -> CreditCollection(x))"),
        Rule("R3: amount > 400 -> credit collection", 1, lambda x: layout.get(x, "amount")[:, 0] > amount_400,
             formula="Forall x (FineAmount(x) > 400 -> CreditCollection(x))"),
        Rule("E1: vehicle class A -> credit collection", 1, lambda x: occurs(layout.get(x, "vehicleClass"), vehicle_a),
             split=False, formula="Forall x (VehicleClass(x) = A -> CreditCollection(x))"),
    ]


RULES = {"bpi12": bpi12_rules, "bpi17": bpi17_rules, "sepsis": sepsis_rules, "traffic": traffic_rules}


def build_knowledge(data):
    return Knowledge(RULES[data.name](data))
