'''Test metrics, computed in the same way for every experiment.'''
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

METRICS = ["accuracy", "f1", "precision", "recall", "compliance"]


def ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


@torch.no_grad()
def evaluate(model, loader, knowledge, model_input, device):
    '''
    accuracy, macro precision / recall / F1 of the predictions (threshold 0.5) and
        compliance: among the (rule, prefix) pairs where a rule R1-R3 fires and the true label agrees with it,
                    the fraction where the prediction agrees too (see nesy/knowledge.py)
        rule_compliance: the same for each rule, the additional rules E1, ... included
    model_input: x -> input of the model (nesy/experiments.ModelInput)
    '''
    model.eval()
    y_true, y_pred, truth = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        predictions = (model(model_input(x)).squeeze(1) > 0.5).long()
        y_true.append(y.long().cpu())
        y_pred.append(predictions.cpu())
        truth.append(knowledge.truth(x).cpu())
    y_true, y_pred, truth = torch.cat(y_true), torch.cat(y_pred), torch.cat(truth)

    rule_labels = knowledge.labels()
    applicable = truth & (y_true.unsqueeze(1) == rule_labels)
    satisfied = applicable & (y_pred.unsqueeze(1) == rule_labels)
    split = torch.tensor([rule.split for rule in knowledge.rules], dtype=torch.bool)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "compliance": ratio(satisfied[:, split].sum(), applicable[:, split].sum()),
        "rule_compliance": {rule.name: ratio(satisfied[:, k].sum(), applicable[:, k].sum())
                            for k, rule in enumerate(knowledge.rules)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
