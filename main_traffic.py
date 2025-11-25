import pandas as pd
import ltn
import torch
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
from model.lstm import LSTMModel
from model.transformer import EventTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import statistics
from metrics import compute_accuracy, compute_metrics, compute_accuracy_a, compute_metrics_fa
from collections import defaultdict, Counter
from data import preprocess_traffic
from data.dataset import NeSyDataset, ModelConfig
import matplotlib.pyplot as plt
import seaborn as sns
import math

import argparse

import warnings
warnings.filterwarnings("ignore")

metrics = defaultdict(list)

dataset = "traffic_fines"

classes = ["Repaid", "Send for credit collection"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--num_epochs_nesy", type=int, default=50, help="Number of epochs for training LTN model")
    parser.add_argument("--model_type", type=str, default="transformer", help="Type of model: lstm or transformer")
    parser.add_argument("--train_vanilla", type=bool, default=True, help="Train vanilla model")
    parser.add_argument("--train_nesy", type=bool, default=True, help="Train LTN model")
    parser.add_argument("--setting", type=str, default="compliance", help="Setting for the experiment (compliance or temporal)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()

args = get_args()

sequence_length = 10

config = ModelConfig(
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout_rate=args.dropout_rate,
    num_epochs = args.num_epochs,
    dataset = "traffic",
    sequence_length = sequence_length,
    seed = args.seed
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("-- Reading dataset")
data = pd.read_csv("data_processed/"+dataset+".csv", dtype={"org:resource": str})

(X_train, y_train, X_val, y_val, X_test, y_test, feature_names), vocab_sizes, scalers = preprocess_traffic.preprocess_eventlog(data, args.seed, args.setting)

print("--- Label distribution")
print("--- Training set")
counts = Counter(y_train)
print(counts)
print("--- Test set")
counts = Counter(y_test)
print(counts)

print(feature_names)

train_dataset = NeSyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = NeSyDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataset = NeSyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

if args.train_vanilla:
    # LSTM BASELINE
    if args.model_type == "transformer":
        model = EventTransformer(vocab_sizes, config, feature_names, model_dim=128, num_classes=1, max_len=config.sequence_length, num_layers=1, num_heads=2, dropout=0.1).to(device)
    else:
        model = LSTMModel(vocab_sizes, config, 1, feature_names).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    training_losses = []
    validation_losses = []
    for epoch in range(config.num_epochs):
        train_losses = []
        for enum, (x, y) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.squeeze(1).cpu(), y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {statistics.mean(train_losses)}")
        training_losses.append(statistics.mean(train_losses))
        model.eval()
        val_losses = []
        for enum, (x, y) in enumerate(val_loader):
            with torch.no_grad():
                x = x.to(device)
                output = model(x)
                loss = criterion(output.squeeze(1).cpu(), y)
                val_losses.append(loss.item())
        print(f"Validation Loss: {statistics.mean(val_losses)}")
        validation_losses.append(statistics.mean(val_losses))
        if epoch >= 5:
            if validation_losses[-1] > validation_losses[-2]:
                print("Validation loss increased, stopping training")
                break
        lstm.train()

    model.eval()
    y_pred = []
    y_true = []

    for enum, (x, y) in enumerate(test_loader):
        with torch.no_grad():
            x = x.to(device)
            outputs = model(x).detach().cpu().numpy()
            predictions = np.where(outputs > 0.5, 1., 0.).flatten()
            for i in range(len(y)):
                y_pred.append(predictions[i])
                y_true.append(y[i].cpu())

    print("Metrics LSTM")
    accuracy = accuracy_score(y_true, y_pred)
    metrics_lstm.append(accuracy)
    print("Accuracy:", accuracy)
    f1 = f1_score(y_true, y_pred, average='macro')
    metrics_lstm.append(f1)
    print("F1 Score:", f1)
    precision = precision_score(y_true, y_pred, average='macro')
    metrics_lstm.append(precision)
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred, average='macro')
    metrics_lstm.append(recall)
    print("Recall:", recall)

if args.model_type == "transformer":
    model = EventTransformer(vocab_sizes, config, feature_names, model_dim=128, num_classes=1, max_len=config.sequence_length, num_layers=1, num_heads=2, dropout=0.1).to(device)
else:
    model = LSTMModel(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(model).to(device)

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel() > 0:
            formulas.extend([
                Forall(x_P, P(x_P))
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

model.eval()
print("Metrics LTN w/o rules")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, model, device, "ltn", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn.append(f1score)
print("Precision:", precision)
metrics_ltn.append(precision)
print("Recall:", recall)
metrics_ltn.append(recall)
print("Compliance:", compliance)
metrics_ltn.append(compliance)

if args.model_type == "transformer":
    model = EventTransformer(vocab_sizes, config, feature_names, model_dim=128, num_classes=1, max_len=config.sequence_length, num_layers=1, num_heads=2, dropout=0.1).to(device)
else:
    model = LSTMModel(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(model).to(device)

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

def create_context_vector(antecedent, consequent, x):
    mean_satisfaction = torch.mean(Implies(antecedent, consequent).value.unsqueeze(1))
    satisfaction_variance = torch.var(Implies(antecedent, consequent).value.unsqueeze(1))
    antecedent_coverage = torch.mean((antecedent.value > 0.5).float()).unsqueeze(0)
    confidence = torch.sum(antecedent.value * consequent.value) / (torch.sum(antecedent.value) + 1e-6)
    context_vector = torch.cat((mean_satisfaction.unsqueeze(0), satisfaction_variance.unsqueeze(0), antecedent_coverage, confidence.unsqueeze(0)), dim=0)
    gating_score = mean_satisfaction.item()*math.exp(-2*satisfaction_variance.item())
    gating_score = max(0.0, min(1.0, gating_score))
    return context_vector, gating_score

payment_less_fine = ltn.Function(func = lambda x: ((x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])))
check_add_penality = ltn.Function(func = lambda x: (x[:, :10] == 1).any(dim=1))
amount_greater_than_400 = ltn.Function(func = lambda x: (x[:, 90:100] > scalers["amount"].transform([[400]])[0][0]).any(dim=1))
vehicle_class_A = ltn.Function(func = lambda x: (x[:, 30:40] == 2).any(dim=1))

for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        formulas.extend([
            Forall(x_All, Implies(check_add_penality(x_All), P(x_All))),
            Forall(x_All, Implies(payment_less_fine(x_All), P(x_All))),
            Forall(x_All, Implies(amount_greater_than_400(x_All), P(x_All))),
            Forall(x_All, Implies(vehicle_class_A(x_All), P(x_All))),
        ])
        sat_agg = SatAgg(*formulas)
        loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))

model.eval()
print("Metrics LTN w/o pruning")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, model, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_B.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_B.append(f1score)
print("Precision:", precision)
metrics_ltn_B.append(precision)
print("Recall:", recall)
metrics_ltn_B.append(recall)
print("Compliance:", compliance)
metrics_ltn_B.append(compliance)

if args.model_type == "transformer":
    model = EventTransformer(vocab_sizes, config, feature_names, model_dim=128, num_classes=1, max_len=config.sequence_length, num_layers=1, num_heads=2, dropout=0.1).to(device)
else:
    model = LSTMModel(vocab_sizes, config, 1, feature_names)
P = ltn.Predicate(model).to(device)

# Knowledge Theory
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

SatAgg = ltn.fuzzy_ops.SatAgg()
params = list(P.parameters())
optimizer = torch.optim.Adam(params, lr=config.learning_rate)

def create_context_vector(antecedent, consequent, x):
    mean_satisfaction = torch.mean(Implies(antecedent, consequent).value.unsqueeze(1))
    satisfaction_variance = torch.var(Implies(antecedent, consequent).value.unsqueeze(1))
    antecedent_coverage = torch.mean((antecedent.value > 0.5).float()).unsqueeze(0)
    confidence = torch.sum(antecedent.value * consequent.value) / (torch.sum(antecedent.value) + 1e-6)
    context_vector = torch.cat((mean_satisfaction.unsqueeze(0), satisfaction_variance.unsqueeze(0), antecedent_coverage, confidence.unsqueeze(0)), dim=0)
    gating_score = mean_satisfaction.item()*math.exp(-2*satisfaction_variance.item())
    gating_score = max(0.0, min(1.0, gating_score))
    return context_vector, gating_score

gat_r1, gat_r2, gat_r3, gat_r4, gat_r5, gat_r6, gat_r7 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
sat_r1, sat_r2, sat_r3, sat_r4, sat_r5, sat_r6, sat_r7, sat_main, sat_main_neg = [], [], [], [], [], [], [], [], []
threshold = 0.6
payment_less_fine = ltn.Function(func = lambda x: ((x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])))
check_add_penality = ltn.Function(func = lambda x: (x[:, :10] == 1).any(dim=1))
amount_greater_than_400 = ltn.Function(func = lambda x: (x[:, 90:100] > scalers["amount"].transform([[400]])[0][0]).any(dim=1))
vehicle_class_A = ltn.Function(func = lambda x: (x[:, 30:40] == 2).any(dim=1))

max_f1_val = 0.0
for epoch in range(args.num_epochs_nesy):
    train_loss = 0.0
    for enum, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x_P = ltn.Variable("x_P", x[y==1])
        x_not_P = ltn.Variable("x_not_P", x[y==0])
        x_All = ltn.Variable("x_All", x)
        formulas = []
        formulas_knowledge = []
        if x_P.value.numel()>0:
            formulas.extend([
                Forall(x_P, P(x_P)),
            ])
        if x_not_P.value.numel()>0:
            formulas.extend([
                Forall(x_not_P, Not(P(x_not_P)))
            ])
        if epoch <= 1:
            formulas_knowledge.extend([
                Forall(x_All, Implies(check_add_penality(x_All), P(x_All))),
                Forall(x_All, Implies(payment_less_fine(x_All), P(x_All))),
                Forall(x_All, Implies(amount_greater_than_400(x_All), P(x_All))),
                Forall(x_All, Implies(vehicle_class_A(x_All), P(x_All))),
            ])
        else:
            if gat_r1 >= threshold:
                formulas.extend([
                    Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, :10] == 1).any(dim=1)),
                    Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: ((x.value[:, :10] == 7).any(dim=1) & (x.value[:, 100:110].max(dim=1).values < x.value[:, 90]))),
                    Forall(x_All, P(x_All), cond_vars=[x_All], cond_fn = lambda x: (x.value[:, 90] > scalers["amount"].transform([[400]])[0][0])),
                    Forall(x_All, Implies(check_add_penality(x_All), P(x_All))),
                ])
            if gat_r2 >= threshold:
                formulas.extend([
                    Forall(x_All, Implies(payment_less_fine(x_All), P(x_All))),
                ])
            if gat_r3 >= threshold:
                formulas.extend([
                    Forall(x_All, Implies(amount_greater_than_400(x_All), P(x_All))),
                ])
            if gat_r4 >= threshold:
                formulas.extend([
                    Forall(x_All, Implies(vehicle_class_A(x_All), P(x_All))),
                ])
        if epoch == 5:
            with torch.no_grad():
                sat_r1 += check_add_penality(x_All).value
                sat_r2 += payment_less_fine(x_All).value
                sat_r3 += amount_greater_than_400(x_All).value
                sat_r4 += vehicle_class_A(x_All).value
                sat_main += P(x_All).value
        sat_agg = SatAgg(*formulas)
        if len(formulas_knowledge) > 0:
            sat_agg_knowledge = SatAgg(*formulas_knowledge)
            loss = 1 - (0.8*sat_agg + 0.2*sat_agg_knowledge)
        else:
            loss = 1 - sat_agg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        del x_P, x_not_P, sat_agg
    train_loss = train_loss / len(train_loader)
    _, f1score, _, _, _ =compute_metrics(val_loader, model, device, "nesy", scalers, dataset)
    if f1score > max_f1_val:
        max_f1_val = f1score
        torch.save(model.state_dict(), "best_model.pth")
        count_early_stop = 0
    print(f1score)
    print(" epoch %d | loss %.4f"
                %(epoch, train_loss))
    if epoch == 5:
        sat_r1 = torch.stack(sat_r1)
        sat_r2 = torch.stack(sat_r2)
        sat_r3 = torch.stack(sat_r3)
        sat_r4 = torch.stack(sat_r4)
        sat_main = torch.stack(sat_main)
        sat_r1 = ltn.LTNObject(sat_r1, ["x_All"])
        sat_r2 = ltn.LTNObject(sat_r2, ["x_All"])
        sat_r3 = ltn.LTNObject(sat_r3, ["x_All"])
        sat_r4 = ltn.LTNObject(sat_r4, ["x_All"])
        sat_main = ltn.LTNObject(sat_main, ["x_All"])
        c1, gat_r1 = create_context_vector(sat_r1, sat_main, x_All)
        c2, gat_r2 = create_context_vector(sat_r2, sat_main, x_All)
        c3, gat_r3 = create_context_vector(sat_r3, sat_main, x_All)
        c4, gat_r4 = create_context_vector(sat_r4, sat_main, x_All)
        
        print("Gating scores after epoch 1:")
        print("Rule 1 gating score:", gat_r1)
        print("Rule 2 gating score:", gat_r2)
        print("Rule 3 gating score:", gat_r3)
        print("Rule 4 gating score:", gat_r4)

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
print("Metrics LTN w pruning")
accuracy, f1score, precision, recall, compliance = compute_metrics(test_loader, model, device, "ltn_w_k", scalers, dataset)
print("Accuracy:", accuracy)
metrics_ltn_B.append(accuracy)
print("F1 Score:", f1score)
metrics_ltn_B.append(f1score)
print("Precision:", precision)
metrics_ltn_B.append(precision)
print("Recall:", recall)
metrics_ltn_B.append(recall)
print("Compliance:", compliance)
metrics_ltn_B.append(compliance)