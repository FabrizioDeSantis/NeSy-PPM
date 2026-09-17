import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import torch
import seaborn as sns

function_ra = lambda x, y: torch.tensor([(9.0 in x[i, :25]) and (6.0 in x[i, 25:50]) and (x[i, -25:] < 400).any() and (y[i] == 1) for i in range(x.shape[0])])
function = lambda arr: (((arr[..., :25] == 9.0).any(dim=-1)) & (~((arr[..., :25] == 9.0) & (arr[..., -25:] != 0.0)).any(dim=-1))).int()

def plot_f1_accuracy(predictions_vanilla, ytrue_vanilla, lengths_vanilla, predictions_nesy, ytrue_nesy, lenghts_nesy, predictions_nesy_w_c, ytrue_nesy_w_c, lenghts_nesy_w_c, dataset_size, mode):

    # --- Calculation ---
    if mode:
        unique_values = sorted(list(set(lengths_vanilla)))  # Get unique and sorted values
    else:
        unique_values = sorted(list(set(lenghts_nesy_w_c)))    
    
    
    f1_scores_vanilla = []
    accuracies_vanilla = []

    f1_scores_nesy = []
    accuracies_nesy = []

    f1_scores_nesy_w_c = []
    accuracies_nesy_w_c = []

    for value in unique_values:
        # Create boolean masks for the current value
        indices = [i for i, x in enumerate(lengths_vanilla) if x == value]
        value_predictions_vanilla = [predictions_vanilla[i] for i in indices]
        value_ytrues_vanilla = [ytrue_vanilla[i] for i in indices]

        # Handle the case where there are no predictions/true labels for a value
        if not value_predictions_vanilla:  # Or len(value_ytrues) == 0, they're the same
            f1_scores_vanilla.append(np.nan)  # Append NaN (Not a Number)
            accuracies_vanilla.append(np.nan)
        else:
            f1 = f1_score(value_ytrues_vanilla, value_predictions_vanilla, zero_division=0, average='macro')
            accuracy = accuracy_score(value_ytrues_vanilla, value_predictions_vanilla)
            f1_scores_vanilla.append(f1)
            accuracies_vanilla.append(accuracy)

        #################

        indices = [i for i, x in enumerate(lenghts_nesy) if x == value]
        value_predictions_nesy = [predictions_nesy[i] for i in indices]
        value_ytrues_nesy = [ytrue_nesy[i] for i in indices]

        # Handle the case where there are no predictions/true labels for a value
        if not value_predictions_nesy:  # Or len(value_ytrues) == 0, they're the same
            f1_scores_nesy.append(np.nan)  # Append NaN (Not a Number)
            accuracies_nesy.append(np.nan)
        else:
            f1 = f1_score(value_ytrues_nesy, value_predictions_nesy, zero_division=0, average='macro')
            accuracy = accuracy_score(value_ytrues_nesy, value_predictions_nesy)
            f1_scores_nesy.append(f1)
            accuracies_nesy.append(accuracy)

        #################

        indices = [i for i, x in enumerate(lenghts_nesy_w_c) if x == value]
        value_predictions_nesy_w_c = [predictions_nesy_w_c[i] for i in indices]
        value_ytrues_nesy_w_c = [ytrue_nesy_w_c[i] for i in indices]

        # Handle the case where there are no predictions/true labels for a value
        if not value_predictions_nesy_w_c:  # Or len(value_ytrues) == 0, they're the same
            f1_scores_nesy_w_c.append(np.nan)  # Append NaN (Not a Number)
            accuracies_nesy_w_c.append(np.nan)
        else:
            f1 = f1_score(value_ytrues_nesy_w_c, value_predictions_nesy_w_c, zero_division=0, average='macro')
            accuracy = accuracy_score(value_ytrues_nesy_w_c, value_predictions_nesy_w_c)
            f1_scores_nesy_w_c.append(f1)
            accuracies_nesy_w_c.append(accuracy)

    min_val, max_val = min(unique_values), max(unique_values)
    #xtick_labels = np.arange(min_val, max_val + 1, 4)
    xtick_labels = [val for val in unique_values if val % 5 == 0]

    # --- Plotting F1Score ---
    plt.figure(figsize=(14, 10))  # Adjust figure size for better visualization
    if mode:
        plt.plot(unique_values, f1_scores_vanilla, marker='o', color='#4c72b0', label='F1-Score Vanilla')
        plt.plot(unique_values, f1_scores_nesy, marker='s', color='#dd8452', label='F1-Score NeSy')
        plt.plot(unique_values, f1_scores_nesy_w_c, marker='^', color='#55a868', label='F1-Score NeSy w/ C')

    plt.xlabel('Prefix Length', fontsize=32)
    plt.ylabel('Score', fontsize=32)
    if mode == 1:
        plt.title('F1-Score', fontsize=32)
        plt.legend(fontsize=24)
        plt.grid(True)  # Add grid lines
        plt.xticks(xtick_labels, rotation=45, ha="right", fontsize=32)   #added rotation for better visualization
        plt.yticks(fontsize=32) 
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.savefig("results/" + str(dataset_size) + "/f1score.pdf")
        plt.close() # close the figure to avoid displaying it in environments that don't support display
    elif mode == 2:
        plt.title('F1-Score', fontsize=32)
        plt.legend(fontsize=24)
        plt.grid(True)  # Add grid lines
        plt.xticks(xtick_labels, rotation=45, ha="right", fontsize=32)   #added rotation for better visualization
        plt.yticks(fontsize=32) 
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.savefig("results/" + str(dataset_size) + "/f1score_modified.pdf")
        plt.close() # close the figure to avoid displaying it in environments that don't support display

     # --- Plotting Accuracy ---
    plt.figure(figsize=(14, 10))  # Adjust figure size for better visualization
    if mode:
        plt.plot(unique_values, accuracies_vanilla, marker='o', color='#4c72b0', label='Accuracy Vanilla')
        plt.plot(unique_values, accuracies_nesy, marker='s', color='#dd8452', label='Accuracy NeSy')
        plt.plot(unique_values, accuracies_nesy_w_c, marker='^', color='#55a868', label='Accuracy NeSy w/ C')
    else:
        plt.plot(unique_values, accuracies_vanilla, marker='o', color='#4c72b0', label='Constraint Satisfaction Vanilla')
        plt.plot(unique_values, accuracies_nesy, marker='s', color='#dd8452', label='Constraint Satisfaction NeSy')
        plt.plot(unique_values, accuracies_nesy_w_c, marker='^', color='#55a868', label='Constraint Satisfaction NeSy w/ C')

    plt.xlabel('Prefix Length', fontsize=32)
    plt.ylabel('Score', fontsize=32)
    if mode:
        plt.title('Accuracy', fontsize=32)
    else:
        plt.title('Constraint Satisfaction', fontsize=32)
    plt.legend(fontsize=24)
    plt.grid(True)  # Add grid lines
    print(unique_values)
    plt.xticks(xtick_labels, rotation=45, ha="right", fontsize=32)   #added rotation for better visualization
    plt.yticks(fontsize=32) 
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    if mode == 1:
        plt.savefig("results/" + str(dataset_size) + "/accuracy.pdf")
    elif mode == 2:
        plt.savefig("results/" + str(dataset_size) + "/accuracy_modified.pdf")
    else:
        plt.savefig("results/" + str(dataset_size) + "/constraint_satisfaction.pdf")

    plt.close() # close the figure to avoid displaying it in environments that don't support display

def plot_metrics_rob(predictions_vanilla, ytrue_vanilla, lengths_vanilla, predictions_nesy, ytrue_nesy, lenghts_nesy, predictions_nesy_w_c, ytrue_nesy_w_c, lenghts_nesy_w_c, dataset_size, mode):

    # --- Calculation ---
    if mode:
        unique_values = sorted(list(set(lengths_vanilla)))  # Get unique and sorted values
    else:
        unique_values = sorted(list(set(lenghts_nesy_w_c)))    
    
    
    f1_scores_vanilla = []
    accuracies_vanilla = []

    f1_scores_nesy = []
    accuracies_nesy = []

    f1_scores_nesy_w_c = []
    accuracies_nesy_w_c = []

    for value in unique_values:
        # Create boolean masks for the current value
        indices = [i for i, x in enumerate(lengths_vanilla) if x == value]
        value_predictions_vanilla = [predictions_vanilla[i] for i in indices]
        value_ytrues_vanilla = [ytrue_vanilla[i] for i in indices]

        # Handle the case where there are no predictions/true labels for a value
        if not value_predictions_vanilla:  # Or len(value_ytrues) == 0, they're the same
            f1_scores_vanilla.append(np.nan)  # Append NaN (Not a Number)
            accuracies_vanilla.append(np.nan)
        else:
            f1 = f1_score(value_ytrues_vanilla, value_predictions_vanilla, zero_division=0, average='macro')
            accuracy = accuracy_score(value_ytrues_vanilla, value_predictions_vanilla)
            f1_scores_vanilla.append(f1)
            accuracies_vanilla.append(accuracy)

        #################

        indices = [i for i, x in enumerate(lenghts_nesy) if x == value]
        value_predictions_nesy = [predictions_nesy[i] for i in indices]
        value_ytrues_nesy = [ytrue_nesy[i] for i in indices]

        # Handle the case where there are no predictions/true labels for a value
        if not value_predictions_nesy:  # Or len(value_ytrues) == 0, they're the same
            f1_scores_nesy.append(np.nan)  # Append NaN (Not a Number)
            accuracies_nesy.append(np.nan)
        else:
            f1 = f1_score(value_ytrues_nesy, value_predictions_nesy, zero_division=0, average='macro')
            accuracy = accuracy_score(value_ytrues_nesy, value_predictions_nesy)
            f1_scores_nesy.append(f1)
            accuracies_nesy.append(accuracy)

        #################

        indices = [i for i, x in enumerate(lenghts_nesy_w_c) if x == value]
        value_predictions_nesy_w_c = [predictions_nesy_w_c[i] for i in indices]
        value_ytrues_nesy_w_c = [ytrue_nesy_w_c[i] for i in indices]

        # Handle the case where there are no predictions/true labels for a value
        if not value_predictions_nesy_w_c:  # Or len(value_ytrues) == 0, they're the same
            f1_scores_nesy_w_c.append(np.nan)  # Append NaN (Not a Number)
            accuracies_nesy_w_c.append(np.nan)
        else:
            f1 = f1_score(value_ytrues_nesy_w_c, value_predictions_nesy_w_c, zero_division=0, average='macro')
            accuracy = accuracy_score(value_ytrues_nesy_w_c, value_predictions_nesy_w_c)
            f1_scores_nesy_w_c.append(f1)
            accuracies_nesy_w_c.append(accuracy)

    min_val, max_val = min(unique_values), max(unique_values)
    #xtick_labels = np.arange(min_val, max_val + 1, 4)
    xtick_labels = [val for val in unique_values if val % 5 == 0]

    # --- Plotting F1Score ---
    plt.figure(figsize=(14, 10))  # Adjust figure size for better visualization
    if mode:
        plt.plot(unique_values, f1_scores_vanilla, marker='o', color='#4c72b0', label='F1-Score Vanilla')
        plt.plot(unique_values, f1_scores_nesy, marker='s', color='#dd8452', label='F1-Score NeSy')
        plt.plot(unique_values, f1_scores_nesy_w_c, marker='^', color='#55a868', label='F1-Score NeSy w/ C')

    plt.xlabel('Prefix Length', fontsize=32)
    plt.ylabel('Score', fontsize=32)
    if mode == 1:
        plt.title('F1-Score', fontsize=32)
        plt.legend(fontsize=24)
        plt.grid(True)  # Add grid lines
        plt.xticks(xtick_labels, rotation=45, ha="right", fontsize=32)   #added rotation for better visualization
        plt.yticks(fontsize=32) 
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.savefig("results/robustness/" + str(dataset_size) + "/f1score.pdf")
        plt.close() # close the figure to avoid displaying it in environments that don't support display
    elif mode == 2:
        plt.title('F1-Score', fontsize=32)
        plt.legend(fontsize=24)
        plt.grid(True)  # Add grid lines
        plt.xticks(xtick_labels, rotation=45, ha="right", fontsize=32)   #added rotation for better visualization
        plt.yticks(fontsize=32) 
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.savefig("results/robustness/" + str(dataset_size) + "/f1score_modified.pdf")
        plt.close() # close the figure to avoid displaying it in environments that don't support display

     # --- Plotting Accuracy ---
    plt.figure(figsize=(14, 10))  # Adjust figure size for better visualization
    if mode:
        plt.plot(unique_values, accuracies_vanilla, marker='o', color='#4c72b0', label='Accuracy Vanilla')
        plt.plot(unique_values, accuracies_nesy, marker='s', color='#dd8452', label='Accuracy NeSy')
        plt.plot(unique_values, accuracies_nesy_w_c, marker='^', color='#55a868', label='Accuracy NeSy w/ C')
    else:
        plt.plot(unique_values, accuracies_vanilla, marker='o', color='#4c72b0', label='Constraint Satisfaction Vanilla')
        plt.plot(unique_values, accuracies_nesy, marker='s', color='#dd8452', label='Constraint Satisfaction NeSy')
        plt.plot(unique_values, accuracies_nesy_w_c, marker='^', color='#55a868', label='Constraint Satisfaction NeSy w/ C')

    plt.xlabel('Prefix Length', fontsize=32)
    plt.ylabel('Score', fontsize=32)
    if mode:
        plt.title('Accuracy', fontsize=32)
    else:
        plt.title('Constraint Satisfaction', fontsize=32)
    plt.legend(fontsize=24)
    plt.grid(True)  # Add grid lines
    print(unique_values)
    plt.xticks(xtick_labels, rotation=45, ha="right", fontsize=32)   #added rotation for better visualization
    plt.yticks(fontsize=32) 
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    if mode == 1:
        plt.savefig("results/robustness/" + str(dataset_size) + "/accuracy.pdf")
    elif mode == 2:
        plt.savefig("results/robustness/" + str(dataset_size) + "/accuracy_modified.pdf")
    else:
        plt.savefig("results/robustness/" + str(dataset_size) + "/constraint_satisfaction.pdf")

    plt.close() # close the figure to avoid displaying it in environments that don't support display

def plot_confusion_matrix(cm, classes, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save figure

def compute_accuracy(loader, model, device):
    mean_accuracy = 0.0
    for data, labels, labels_t, lengths, cases in loader:
        data = data.to(device)
        predictions = model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        mean_accuracy += accuracy_score(labels, predictions)
    return mean_accuracy / len(loader)

def compute_accuracy_test_rob(loader, model, mode, device):
    mean_accuracy = 0.0
    num_constraints = 0
    correct_constraints = 0
    y_true = []
    y_true_modified = []
    y_pred = []
    case_ids = []
    lengths = []
    true_constraints = []
    predicted_constraints = []
    lengths_constraints = []

    count_temp = 0
    count_temp2 = 0
    for data, labels, y_ra, l, cases in loader:
        #constraints = function(data).detach().cpu().numpy()
        constraints = function_ra(data, y_ra).detach().cpu().numpy()
        with torch.no_grad():
            data = data.to(device)
            predictions = model(data).detach().cpu().numpy()
            predictions = np.where(predictions > 0.5, 1., 0.).flatten()
            mean_accuracy += accuracy_score(labels, predictions)
            for i in range(len(labels)):
                y_true.append(labels[i])
                y_pred.append(predictions[i])
                case_ids.append(cases[i])
                if constraints[i] == 1:
                    true_constraints.append(1)
                    y_true_modified.append(0.)
                    lengths_constraints.append(l[i].item())
                    num_constraints += 1
                    if predictions[i] != 1:
                        correct_constraints += 1
                        predicted_constraints.append(1)
                    else:
                        predicted_constraints.append(0)
                else:
                    y_true_modified.append(labels[i])
            lengths.extend(l.tolist())
    if mode == 0:
        compliance = None
        comp_dict = None
    else:
        compliance = 0
        comp_dict = 0

    cm = confusion_matrix(y_true, y_pred)
    cm_modified = confusion_matrix(y_true_modified, y_pred)
    if num_constraints == 0:
        print("No constraints found in the test set.")
        compliance = -1
    else:
        compliance = correct_constraints / num_constraints
    #print("Compliance: ", correct_constraints / num_constraints)

    return compliance, mean_accuracy / len(loader), f1_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), cm, y_true, y_pred, lengths, true_constraints, predicted_constraints, lengths_constraints, y_true_modified, (accuracy_score(y_true_modified, y_pred), f1_score(y_true_modified, y_pred, average='macro'), precision_score(y_true_modified, y_pred, average='macro'), recall_score(y_true_modified, y_pred, average='macro'), cm_modified)

def compute_accuracy_test(loader, model, mode, device):
    mean_accuracy = 0.0
    num_constraints = 0
    correct_constraints = 0
    y_true = []
    y_true_modified = []
    y_pred = []
    case_ids = []
    lengths = []
    true_constraints = []
    predicted_constraints = []
    lengths_constraints = []

    count_temp = 0
    count_temp2 = 0
    for data, labels, y_ra, l, cases in loader:
        constraints = function(data).detach().cpu().numpy()
        #constraints = function_ra(data, y_ra).detach().cpu().numpy()
        with torch.no_grad():
            data = data.to(device)
            predictions = model(data).detach().cpu().numpy()
            predictions = np.where(predictions > 0.5, 1., 0.).flatten()
            mean_accuracy += accuracy_score(labels, predictions)
            for i in range(len(labels)):
                y_true.append(labels[i])
                y_pred.append(predictions[i])
                case_ids.append(cases[i])
                if constraints[i] == 1:
                    true_constraints.append(1)
                    y_true_modified.append(0.)
                    lengths_constraints.append(l[i].item())
                    num_constraints += 1
                    if predictions[i] != 1:
                        correct_constraints += 1
                        predicted_constraints.append(1)
                    else:
                        predicted_constraints.append(0)
                else:
                    y_true_modified.append(labels[i])
            lengths.extend(l.tolist())
    if mode == 0:
        compliance = None
        comp_dict = None
    else:
        compliance = 0
        comp_dict = 0

    cm = confusion_matrix(y_true, y_pred)
    cm_modified = confusion_matrix(y_true_modified, y_pred)
    if num_constraints == 0:
        print("No constraints found in the test set.")
        compliance = -1
    else:
        compliance = correct_constraints / num_constraints
    #print("Compliance: ", correct_constraints / num_constraints)

    return compliance, mean_accuracy / len(loader), f1_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), cm, y_true, y_pred, lengths, true_constraints, predicted_constraints, lengths_constraints, y_true_modified, (accuracy_score(y_true_modified, y_pred), f1_score(y_true_modified, y_pred, average='macro'), precision_score(y_true_modified, y_pred, average='macro'), recall_score(y_true_modified, y_pred, average='macro'), cm_modified)

def compute_accuracy(loader, model, device):
    y_pred = []
    y_true = []
    for data, labels in loader:
        data = data.to(device)
        predictions = model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        for i in range(len(labels)):
            y_pred.append(predictions[i])
            y_true.append(labels[i].cpu())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def compute_accuracy_a(loader, model, device, rule_1, rule_2, rule_3):
    y_pred = []
    y_true = []
    for data, labels in loader:
        data = data.to(device)
        predictions = model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        for i in range(len(labels)):
            y_pred.append(predictions[i])
            y_true.append(labels[i].cpu())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def compute_metrics_fa(loader, model, device, mode, scalers, dataset):
    ### SEPSIS RULES
    if dataset == "sepsis":
        rule_lact = lambda x: (x[..., 351:364] > scalers["LacticAcid"].transform([[2]])[0][0]).any(dim=1)
        rule_crit = lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1))
        rule_crp_atb = lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x]).to(device)
        rule_crp_100 = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1)
    ### BPI12 RULES
    if dataset == "bpi12":
        rule_amount_1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
        rule_amount_2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
        rule_amount_3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
        rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
        rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)
    # TRAFFIC RULES
    if dataset == "traffic_fines":
        rule_penalty = lambda x: (x[:, :10] == 1).any(dim=1)
        rule_payment = lambda x: (x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])
        rule_amount = lambda x: (x[:, 90] > scalers["amount"].transform([[400]])[0][0])
    ###########
    # BPI17 RULES
    if dataset == "bpi17":
        rule_1 = lambda x: ((x[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
        rule_2 = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1)
        rule_3 = lambda x: (x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, 20:40] == 6).any(dim=1)
    y_pred = []
    y_true = []
    compliance = 0
    satisfied_constraints = 0
    num_constraints = 0
    for data, labels in loader:
        data = data.to(device)
        if dataset == "bpi17":
            rule_1_res = rule_1(data).detach()
            rule_2_res = rule_2(data).detach()
            rule_3_res = rule_3(data).detach()
            data = torch.cat([data, rule_1_res.unsqueeze(1).repeat(1, 20)], dim=1)
            data = torch.cat([data, rule_2_res.unsqueeze(1).repeat(1, 20)], dim=1)
            data = torch.cat([data, rule_3_res.unsqueeze(1).repeat(1, 20)], dim=1)
        if dataset == "sepsis":
            rule_lact_res = rule_lact(data).detach()
            rule_crit_res= rule_crit(data).detach()
            rule_crp_atb_res = rule_crp_atb(data).detach()
            rule_crp_100_res = rule_crp_100(data).detach()
            rule_3_res = torch.logical_and(rule_crp_atb_res, rule_crp_100_res).detach()
            data = torch.cat([data, rule_lact_res.unsqueeze(1).repeat(1, 13)], dim=1)
            data = torch.cat([data, rule_crit_res.unsqueeze(1).repeat(1, 13)], dim=1)
            data = torch.cat([data, rule_3_res.unsqueeze(1).repeat(1, 13)], dim=1)
        elif dataset == "bpi12":
            rule_amount_1_res = rule_amount_1(data).detach()
            rule_amount_2_res = rule_amount_2(data).detach()
            rule_amount_3_res = rule_amount_3(data).detach()
            rule_resource_1_res = rule_resource_1(data).detach()
            rule_resource_2_res = rule_resource_2(data).detach()
            rule_2_res = torch.logical_and(rule_amount_2_res, rule_amount_3_res).detach()
            rule_3_res = torch.logical_or(rule_resource_1_res, rule_resource_2_res).detach()
            data = torch.cat([data, rule_amount_1_res.unsqueeze(1).repeat(1, 40)], dim=1)
            data = torch.cat([data, rule_2_res.unsqueeze(1).repeat(1, 40)], dim=1)
            data = torch.cat([data, rule_3_res.unsqueeze(1).repeat(1, 40)], dim=1)
        elif dataset == "traffic_fines":
            rule_penalty_res = rule_penalty(data).detach()
            rule_payment_res = rule_payment(data).detach()
            rule_amount_res = rule_amount(data).detach()
            data = torch.cat([data, rule_penalty_res.unsqueeze(1).repeat(1, 10)], dim=1)
            data = torch.cat([data, rule_payment_res.unsqueeze(1).repeat(1, 10)], dim=1)
            data = torch.cat([data, rule_amount_res.unsqueeze(1).repeat(1, 10)], dim=1)
        predictions = model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        for i in range(len(labels)):
            y_pred.append(predictions[i])
            y_true.append(labels[i].cpu())
            if dataset == "bpi12":
                if rule_amount_1_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_amount_2_res[i] == 1 and rule_amount_3_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_resource_1_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_resource_2_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
            elif dataset == "sepsis":
                if rule_crit_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_crp_atb_res[i] == 1 and rule_crp_100_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
            elif dataset == "traffic_fines":
                if rule_penalty_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_payment_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1         
                if rule_amount_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
            if dataset == "bpi17":
                if rule_1_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_2_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_3_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1   
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    compliance = satisfied_constraints / num_constraints
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if mode == "ltn":
        plt.savefig("confusion_matrix_ltn.png", dpi=300, bbox_inches='tight')
    elif mode == "ltn_w_k":
        plt.savefig("confusion_matrix_ltn_w_k.png", dpi=300, bbox_inches='tight')  # Save figure
    plt.close()  # Close the figure to avoid displaying it in environments that don't support display
    return accuracy, f1score, precision, recall, compliance

def compute_metrics(loader, model, device, mode, scalers, dataset):
    ### SEPSIS RULES
    if dataset == "sepsis":
        rule_1 = lambda x: (x[..., 351:364] > scalers["LacticAcid"].transform([[4]])[0][0]).any(dim=1)
        rule_2 = lambda x: (x[:, :13].eq(1).any(dim=1)) & (x[:, 39:52].eq(1).any(dim=1)) & (x[:, 65:78].eq(1).any(dim=1))
        rule_crp_atb = lambda x: torch.tensor([int(any(i < j for i in (row[104:117] == 2).nonzero(as_tuple=True)[0] for j in (row[104:117] == 6).nonzero(as_tuple=True)[0])) for row in x]).to(device)
        rule_crp_100 = lambda x: (x[:, 338:351] > scalers["CRP"].transform([[100]])[0][0]).any(dim=1)
    # BPI12 RULES
    if dataset == "bpi12":
        rule_amount_1 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[10000]])[0][0]).any(dim=1)
        rule_amount_2 = lambda x: (x[:, 200:240] > scalers["case:AMOUNT_REQ"].transform([[50000]])[0][0]).any(dim=1)
        rule_amount_3 = lambda x: (x[:, 200:240] < scalers["case:AMOUNT_REQ"].transform([[60000]])[0][0]).any(dim=1)
        rule_resource_1 = lambda x: (x[:, :240] == 48).any(dim=1)
        rule_resource_2 = lambda x: (x[:, :240] == 21).any(dim=1)
    # TRAFFIC RULES
    if dataset == "traffic_fines":
        rule_penalty = lambda x: (x[:, :10] == 1).any(dim=1)
        rule_payment = lambda x: (x[:, :10] == 7).any(dim=1) & (x[:, 100:110].max(dim=1).values < x[:, 90])
        rule_amount = lambda x: (x[:, 90] > scalers["amount"].transform([[400]])[0][0])
    # BPI17 RULES
    if dataset == "bpi17":
        rule_1 = lambda x: ((x[:, 140] < scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, :20] == 11).any(dim=1) & (x[:, 40:60] != 0).any(dim=1))
        rule_2 = lambda x: (x[:, 40:60] == 0).all(dim=1) & (x[:, :20] == 11).any(dim=1)
        rule_3 = lambda x: (x[:, 140] > scalers["case:RequestedAmount"].transform([[20000]])[0][0]) & (x[:, 20:40] == 6).any(dim=1)
    ############
    y_pred = []
    y_true = []
    compliance = 0
    satisfied_constraints = 0
    num_constraints = 0
    for data, labels in loader:
        data = data.to(device)
        predictions = model(data).detach().cpu().numpy()
        predictions = np.where(predictions > 0.5, 1., 0.).flatten()
        if dataset == "sepsis":
            rule_1_res = rule_1(data).detach().cpu().numpy()
            rule_2_res = rule_2(data).detach().cpu().numpy()
            rule_crp_atb_res = rule_crp_atb(data).detach().cpu().numpy()
            rule_crp_100_res = rule_crp_100(data).detach().cpu().numpy()
        elif dataset == "bpi12":
            rule_amount_1_res = rule_amount_1(data).detach().cpu().numpy()
            rule_amount_2_res = rule_amount_2(data).detach().cpu().numpy()
            rule_amount_3_res = rule_amount_3(data).detach().cpu().numpy()
            rule_resource_1_res = rule_resource_1(data).detach().cpu().numpy()
            rule_resource_2_res = rule_resource_2(data).detach().cpu().numpy()
        elif dataset == "traffic_fines":
            rule_penalty_res = rule_penalty(data).detach().cpu().numpy()
            rule_payment_res = rule_payment(data).detach().cpu().numpy()
            rule_amount_res = rule_amount(data).detach().cpu().numpy()
        elif dataset == "bpi17":
            rule_1_res = rule_1(data).detach().cpu().numpy()
            rule_2_res = rule_2(data).detach().cpu().numpy()
            rule_3_res = rule_3(data).detach().cpu().numpy()
        for i in range(len(labels)):
            y_pred.append(predictions[i])
            y_true.append(labels[i].cpu())
            if dataset == "bpi12":
                if rule_amount_1_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_amount_2_res[i] == 1 and rule_amount_3_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_resource_1_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_resource_2_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
            elif dataset == "sepsis":
                if rule_1_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_2_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_crp_atb_res[i] == 1 and rule_crp_100_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
            elif dataset == "traffic_fines":
                if rule_penalty_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_payment_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1         
                if rule_amount_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1   
            elif dataset == "bpi17":
                if rule_1_res[i] == 1 and labels[i] == 1:
                    num_constraints += 1
                    if predictions[i] == 1:
                        satisfied_constraints += 1
                if rule_2_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
                if rule_3_res[i] == 1 and labels[i] == 0:
                    num_constraints += 1
                    if predictions[i] == 0:
                        satisfied_constraints += 1
    accuracy = accuracy_score(y_true, y_pred)
    f1score = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    compliance = satisfied_constraints / num_constraints
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if mode == "ltn":
        plt.savefig("confusion_matrix_ltn.png", dpi=300, bbox_inches='tight')
    elif mode == "ltn_w_k":
        plt.savefig("confusion_matrix_ltn_w_k.png", dpi=300, bbox_inches='tight')  # Save figure
    elif mode == "nesy":
        plt.savefig("confusion_matrix_nesy.png", dpi=300, bbox_inches='tight')  # Save figure
    plt.close()  # Close the figure to avoid displaying it in environments that don't support display
    return accuracy, f1score, precision, recall, compliance