from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data, seed, ratio=0.2):
    random.seed(seed)
    grouped = data.groupby("case:concept:name")
    unique_groups = list(grouped.groups.keys())  # Get the unique group IDs
    labels = data.groupby("case:concept:name")["label"].first().to_list()

    len_test_set = int(len(unique_groups) * ratio)

    labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
    labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
    labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()
    labels_r4 = data.groupby("case:concept:name")["rule_4"].first().to_list()
    labels_r5 = data.groupby("case:concept:name")["rule_5"].first().to_list()

    filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 0)]
    filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 0)]
    filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 0)]
    filtered_values_r4 = [v for v, l, r in zip(unique_groups, labels, labels_r4) if (r == 1 and l == 0)]
    filtered_values_r5 = [v for v, l, r in zip(unique_groups, labels, labels_r5) if (r == 1 and l == 0)]

    filtered_values_nor = [v for v, l, r1, r2, r3, r4, r5 in zip(unique_groups, labels, labels_r1, labels_r2, labels_r3, labels_r4, labels_r5) if (r1 == 0 and r2 == 0 and r3 == 0 and l == 1)]

    filtered_values = filtered_values_r1 + filtered_values_r2 + filtered_values_r3 + filtered_values_nor
    compliant_ids = list(set(filtered_values_r1 + filtered_values_r2 + filtered_values_r3))
    filtered_values = list(set(filtered_values))

    print("Len filtered values: ", len(filtered_values))
    if len(filtered_values) > len_test_set:
        test_ids = random.sample(filtered_values, len_test_set)
    else:
        test_ids = filtered_values

    training_ids = [x for x in unique_groups if x not in test_ids]
    training_ids = list(set(training_ids)) 

    compliant_training_ids = [x for x in training_ids if x in compliant_ids]
    compliant_test_ids = [x for x in test_ids if x in compliant_ids]
    print("Len training set: ", len(training_ids))
    print("Len test set: ", len(test_ids))
    print("Number of compliant traces in training set: ", len(compliant_training_ids))
    print("Number of compliant traces in test set: ", len(compliant_test_ids))

    return training_ids, test_ids

def create_train_val_test_split(data, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2):
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

def create_ngrams(data, train_ids, test_ids, window_size=40):

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

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str", "rule_1", "rule_2", "rule_3", "rule_4", "rule_5"])

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

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "concept:name_str", "rule_1", "rule_2", "rule_3", "rule_4", "rule_5"])
        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    print(feature_names)

    return ngrams_training, labels_training, ngrams_test, labels_test, feature_names

def preprocess_eventlog(data, seed, setting="compliance"):

    vocab_sizes = {}
    cases = data[data["concept:name"] == "A_SUBMITTED-COMPLETE"]
    labels = cases["label"].to_list()
    case_ids = cases["case:concept:name"].to_list()
    print("Number of traces: ", len(labels))

    if setting == "compliance":
        train_ids, test_ids = create_test_set(data, seed)
        data_training = data[data["case:concept:name"].isin(train_ids)]
        train_ids, val_ids = create_test_set(data_training, seed, ratio=0.2)
    else:
        train_ids, val_ids, test_ids = create_train_val_test_split(data, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2)

    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in val set: ", len(val_ids))
    print("Number of traces in test set: ", len(test_ids))

    scaler_ar = MinMaxScaler()
    scaler_elapsed = MinMaxScaler()
    scaler_time_prev = MinMaxScaler()
    
    labels = data.groupby("case:concept:name")["label"].first().reset_index()
    print(data.columns)
    data = data.drop(columns=["case:REG_DATE"])

    data["concept:name_str"] = data["concept:name"]
    print("concept:name_str: ", data["concept:name_str"].unique())
    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("W_Completeren aanvraag-COMPLETE: ", data["concept:name"].cat.categories.get_loc("W_Completeren aanvraag-COMPLETE") + 1)
    print("W_Valideren aanvraag-COMPLETE: ", data["concept:name"].cat.categories.get_loc("W_Valideren aanvraag-COMPLETE") + 1)
    print("W_Nabellen offertes-START: ", data["concept:name"].cat.categories.get_loc("W_Nabellen offertes-START") + 1)
    print("O_SENT_BACK-COMPLETE: ", data["concept:name"].cat.categories.get_loc("O_SENT_BACK-COMPLETE") + 1)
    print("O_CANCELLED-COMPLETE: ", data["concept:name"].cat.categories.get_loc("O_CANCELLED-COMPLETE") + 1)
    print("A_ACCEPTED-COMPLETE: ", data["concept:name"].cat.categories.get_loc("A_ACCEPTED-COMPLETE") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["org:resource"] = pd.Categorical(data["org:resource"])
    res_11169 = data["org:resource"].cat.categories.get_loc("11169") + 1
    print("11169 code: ", res_11169)
    res_10910 = data["org:resource"].cat.categories.get_loc("10910") + 1
    print("10910 code: ", res_10910)
    data["org:resource"] = data["org:resource"].cat.codes + 1
    vocab_sizes["org:resource"] = data["org:resource"].max()

    # Numerical values
    data["case:AMOUNT_REQ"] = data["case:AMOUNT_REQ"].ffill()
    data["case:AMOUNT_REQ"] = scaler_ar.fit_transform(data[["case:AMOUNT_REQ"]])
    data["elapsed_time"] = data["elapsed_time"].fillna(0)
    data["elapsed_time"] = scaler_elapsed.fit_transform(data[["elapsed_time"]])
    data["time_since_previous"] = data["time_since_previous"].fillna(0)
    data["time_since_previous"] = scaler_time_prev.fit_transform(data[["time_since_previous"]])

    scalers = {
        "case:AMOUNT_REQ": scaler_ar,
        "elapsed_time": scaler_elapsed,
        "time_since_previous": scaler_time_prev
    }

    return create_ngrams(data, train_ids, val_ids, test_ids), vocab_sizes, scalers