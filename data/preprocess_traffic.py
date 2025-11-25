from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data, seed, ratio=0.2):
    random.seed(seed)
    grouped = data.groupby("case:concept:name")
    unique_groups = list(grouped.groups.keys())
    labels = data.groupby("case:concept:name")["label"].first().to_list()

    len_test_set = int(len(unique_groups) * ratio)

    labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
    labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
    labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()
    labels_r4 = data.groupby("case:concept:name")["rule_4"].first().to_list()

    filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 1)]
    filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 1)]
    filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 1)]
    filtered_values_r4 = [v for v, l, r in zip(unique_groups, labels, labels_r4) if (r == 1 and l == 1)]

    compliant_ids = list(set(filtered_values_r1 + filtered_values_r2 + filtered_values_r3 + filtered_values_r4))
    filtered_values_nor = [v for v, l, r1, r2, r3, r4 in zip(unique_groups, labels, labels_r1, labels_r2, labels_r3, labels_r4) if (r1 == 0 and r2 == 0 and r3 == 0 and r4 == 0 and l == 0)]
    filtered_values_r1_training = random.sample(filtered_values_r1, len(filtered_values_r1) // 2)
    filtered_values_r1_test = [x for x in filtered_values_r1 if x not in filtered_values_r1_training]
    filtered_values_r2_training = random.sample(filtered_values_r2, len(filtered_values_r2) // 2)
    filtered_values_r2_test = [x for x in filtered_values_r2 if x not in filtered_values_r2_training]
    filtered_values_r3_training = random.sample(filtered_values_r3, len(filtered_values_r3) // 2)
    filtered_values_r3_test = [x for x in filtered_values_r3 if x not in filtered_values_r3_training]
    filtered_values_r4_training = random.sample(filtered_values_r4, len(filtered_values_r4) // 2)
    filtered_values_r4_test = [x for x in filtered_values_r4 if x not in filtered_values_r4_training]

    filtered_values = filtered_values_r1_test + filtered_values_r2_test + filtered_values_r3_test + filtered_values_r4_test + filtered_values_nor
    filtered_values = list(set(filtered_values))

    if len(filtered_values) > len_test_set:
        test_ids = random.sample(filtered_values, len_test_set)
    else:
        test_ids = filtered_values

    training_ids = [x for x in unique_groups if x not in test_ids]
    training_ids = list(set(training_ids))

    compliant_test_ids = [x for x in test_ids if x in compliant_ids]
    compliant_training_ids = [x for x in training_ids if x in compliant_ids]
    print("Number of traces in training set: ", len(training_ids))
    print("Number of traces in test set: ", len(test_ids))
    print("Number of compliant traces in training set: ", len(compliant_training_ids))
    print("Number of compliant traces in test set: ", len(compliant_test_ids))

    return training_ids, test_ids

def create_ngrams(data, train_ids, test_ids, window_size=10):

    ngrams_test = []
    ngrams_training = []
    labels_training = []
    labels_test = []

    training_data = data[data["case:concept:name"].isin(train_ids)]
    test_data = data[data["case:concept:name"].isin(test_ids)]

    for id_value, group in training_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
        label = int(group['label'].dropna().iloc[0])

        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp"])

        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_training.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_training.append(cols)

    for id_value, group in test_data.groupby('case:concept:name'):

        group = group.reset_index(drop=True)
                
        label = int(group['label'].dropna().iloc[0])
        if len(group) > window_size:
            group = group.iloc[:window_size]

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp"])
        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    return ngrams_training, labels_training, ngrams_test, labels_test, feature_names

def create_train_val_test_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.2):
    data = data.sort_values(by=['case:concept:name', 'time:timestamp'])
    graph_ids = data['case:concept:name'].unique()
    num_graphs = len(graph_ids)
    test_size = int(num_graphs * test_ratio)
    val_size = int((num_graphs - test_size) * val_ratio)
    train_size = num_graphs - test_size - val_size
    print(f"Total graphs: {num_graphs}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    train_ids = graph_ids[:train_size]
    val_ids = graph_ids[train_size:train_size + val_size]
    test_ids = graph_ids[train_size + val_size:]
    return train_ids, val_ids, test_ids

def preprocess_eventlog(data, seed, setting="compliance"):

    data = data.dropna(subset=["case:concept:name"])
    vocab_sizes = {}
    cases = data[data["concept:name"] == "Create Fine"]
    labels = cases["label"].to_list()
    case_ids = cases["case:concept:name"].to_list()
    print(len(case_ids))
    print("Number of traces: ", len(labels))

    scalers = {}
    scaler_amount = MinMaxScaler()
    scaler_paymentamount = MinMaxScaler()
    scaler_totalpaymentamount = MinMaxScaler()
    scaler_expense = MinMaxScaler()
    scaler_elapsed = MinMaxScaler()
    scaler_time_prev = MinMaxScaler()

    print(data.columns)

    if setting == "compliance":
        train_ids, test_ids = create_test_set(data, seed)
        data_training = data[data["case:concept:name"].isin(train_ids)]
        train_ids, val_ids = create_test_set(data_training, seed, ratio=0.2)
    else:
        train_ids, val_ids, test_ids = create_train_val_test_split(data, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)

    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in test set: ", len(test_ids))

    data = data.drop(columns=["lifecycle:transition", "matricola", 'totalPaymentAmount'])
    data = data[['case:concept:name', 'concept:name', 'time:timestamp', 'org:resource', 'dismissal', 'vehicleClass',
       'article', 'points', 'expense', 'notificationType',
       'lastSent', 'amount', 'paymentAmount', 'label']]

    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("Add penalty: ", data["concept:name"].cat.categories.get_loc("Add penalty") + 1)
    print("Create fine: ", data["concept:name"].cat.categories.get_loc("Create Fine") + 1)
    print("Send fine: ", data["concept:name"].cat.categories.get_loc("Send Fine") + 1)
    print("Payment: ", data["concept:name"].cat.categories.get_loc("Payment") + 1)
    print("Insert Fine Notification: ", data["concept:name"].cat.categories.get_loc("Insert Fine Notification") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["org:resource"] = data["org:resource"].fillna(0)
    data["org:resource"] = pd.Categorical(data["org:resource"])
    data["org:resource"] = data["org:resource"].cat.codes + 1
    vocab_sizes["org:resource"] = data["org:resource"].max()

    data["dismissal"] = data["dismissal"].ffill()
    data["dismissal"] = pd.Categorical(data["dismissal"])
    data["dismissal"] = data["dismissal"].cat.codes + 1
    vocab_sizes["dismissal"] = data["dismissal"].max()

    data["amount"] = data["amount"].ffill()
    data["amount"] = data["amount"].fillna(0)
    data["amount"] = scaler_amount.fit_transform(data[["amount"]])
    scalers["amount"] = scaler_amount

    data["lastSent"] = data["lastSent"].fillna(0)
    data["lastSent"] = pd.Categorical(data["lastSent"])
    data["lastSent"] = data["lastSent"].cat.codes + 1
    vocab_sizes["lastSent"] = data["lastSent"].max()

    data["paymentAmount"] = data["paymentAmount"].fillna(0)
    scaler_paymentamount = scaler_paymentamount.fit_transform(data[["paymentAmount"]])
    scalers["paymentAmount"] = scaler_paymentamount

    data["points"] = data["points"].fillna(-1)
    data["points"] = pd.Categorical(data["points"])
    data["points"] = data["points"].cat.codes + 1
    vocab_sizes["points"] = data["points"].max()

    data["vehicleClass"] = data["vehicleClass"].fillna(0)
    print("Unique vehicleClass values:", data["vehicleClass"].unique())
    data["vehicleClass"] = pd.Categorical(data["vehicleClass"])
    print("Vehicle class categories:", data["vehicleClass"].cat.categories.get_loc("A") + 1)
    print("Vehicle class categories:", data["vehicleClass"].cat.categories.get_loc("C") + 1)
    print("Vehicle class categories:", data["vehicleClass"].cat.categories.get_loc("M") + 1)
    print("Vehicle class categories:", data["vehicleClass"].cat.categories.get_loc("R") + 1)
    data["vehicleClass"] = data["vehicleClass"].cat.codes + 1
    vocab_sizes["vehicleClass"] = data["vehicleClass"].max()

    data["notificationType"] = data["notificationType"].fillna(0)
    data["notificationType"] = pd.Categorical(data["notificationType"])
    data["notificationType"] = data["notificationType"].cat.codes + 1
    vocab_sizes["notificationType"] = data["notificationType"].max()

    data["expense"] = data["expense"].fillna(0)
    data["expense"] = scaler_expense.fit_transform(data[["expense"]])
    scalers["expense"] = scaler_expense

    data["elapsed_time"] = data["elapsed_time"].fillna(0)
    data["elapsed_time"] = scaler_elapsed.fit_transform(data[["elapsed_time"]])
    scalers["elapsed_time"] = scaler_elapsed

    data["time_since_previous"] = data["time_since_previous"].fillna(0)
    data["time_since_previous"] = scaler_time_prev.fit_transform(data[["time_since_previous"]])
    scalers["time_since_previous"] = scaler_time_prev

    data["article"] = data["article"].ffill()
    data["article"] = pd.Categorical(data["article"])
    data["article"] = data["article"].cat.codes + 1
    vocab_sizes["article"] = data["article"].max()

    return create_ngrams(data, train_ids, val_ids, test_ids), vocab_sizes, scalers