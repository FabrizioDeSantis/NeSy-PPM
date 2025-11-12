from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import random

def create_test_set(data, seed):
    random.seed(seed)
    grouped = data.groupby("case:concept:name")
    unique_groups = list(grouped.groups.keys())
    labels = data.groupby("case:concept:name")["label"].first().to_list()

    len_test_set = int(len(unique_groups) * 0.3)
    len_training_set = len(unique_groups) - len_test_set

    labels_r1 = data.groupby("case:concept:name")["rule_1"].first().to_list()
    labels_r2 = data.groupby("case:concept:name")["rule_2"].first().to_list()
    labels_r3 = data.groupby("case:concept:name")["rule_3"].first().to_list()
    labels_r4 = data.groupby("case:concept:name")["rule_4"].first().to_list()

    filtered_values_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l == 1)]
    filtered_values_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l == 0)]
    filtered_values_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l == 0)]
    filtered_values_r4 = [v for v, l, r in zip(unique_groups, labels, labels_r4) if (r == 1 and l == 0)]
    compliant_ids = list(set(filtered_values_r1 + filtered_values_r2 + filtered_values_r3 + filtered_values_r4))

    filtered_no_r1 = [v for v, l, r in zip(unique_groups, labels, labels_r1) if (r == 1 and l != 1)]
    filtered_no_r2 = [v for v, l, r in zip(unique_groups, labels, labels_r2) if (r == 1 and l != 0)]
    filtered_no_r3 = [v for v, l, r in zip(unique_groups, labels, labels_r3) if (r == 1 and l != 0)]
    filtered_no_r4 = [v for v, l, r in zip(unique_groups, labels, labels_r4) if (r == 1 and l != 0)]
    non_compliant_ids = list(set(filtered_no_r1 + filtered_no_r2 + filtered_no_r3 + filtered_no_r4))

    print("Len training set: ", len_training_set)
    print("Len test set: ", len_test_set)
    print("Number of compliant traces: ", len(compliant_ids))
    print("Number of non-compliant traces: ", len(non_compliant_ids))

    if len(compliant_ids) > len_test_set:
        test_ids = random.sample(compliant_ids, int(len_test_set*0.7))
        remaining_ids = [x for x in unique_groups if x not in test_ids and x not in non_compliant_ids]
        remaining_ids = random.sample(remaining_ids, len_test_set - len(test_ids))
        test_ids = test_ids + remaining_ids
    else:
        test_ids = compliant_ids

    training_ids = [x for x in unique_groups if x not in test_ids]

    print("Compliant traces in training set: ", len([x for x in training_ids if x in compliant_ids]))
    print("Compliant traces in test set: ", len([x for x in test_ids if x in compliant_ids]))

    return training_ids, test_ids

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
    return train_ids, test_ids

def create_ngrams(data, train_ids, test_ids, window_size=20):

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

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "rule_1", "rule_2", "rule_3"])

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

        group = group.drop(columns=["label", "case:concept:name", "time:timestamp", "rule_1", "rule_2", "rule_3"])
        
        feature_names = group.columns.tolist()
        for n in range(1, len(group), 1):
            labels_test.append(label)
            ngram_df = group.iloc[:n]
            list_of_lists = ngram_df.values.tolist()
            cols = [list(col) for col in zip(*list_of_lists)]
            cols = [inner_list + [0] * (window_size-len(inner_list)) for inner_list in cols]
            ngrams_test.append(cols)

    return ngrams_training, labels_training, ngrams_test, labels_test, feature_names

def preprocess_eventlog(data, seed, dataset_size=None, setting="compliance"):
    vocab_sizes = {}
    cases = data[data["concept:name"] == "A_Create Application"]
    labels = cases["label"].to_list()
    case_ids = cases["case:concept:name"].to_list()
    print(len(case_ids))
    print("Number of traces: ", len(labels))

    if setting == "compliance":
        train_ids, test_ids = create_test_set(data, seed)
    else:
        train_ids, test_ids = create_train_val_test_split(data)
    print("Number of traces in train set: ", len(train_ids))
    print("Number of traces in test set: ", len(test_ids))

    scalers = {}

    data = data.drop(columns=["OfferID", "EventID", "EventOrigin", "Accepted", "Selected"])

    print(data.columns)

    data["concept:name"] = pd.Categorical(data["concept:name"])
    print("O_Created code: ", data["concept:name"].cat.categories.get_loc("O_Created") + 1)
    print("O_Create Offer code: ", data["concept:name"].cat.categories.get_loc("O_Create Offer") + 1)
    print("A_Accepted code: ", data["concept:name"].cat.categories.get_loc("A_Accepted") + 1)
    print("O_Create Offer code: ", data["concept:name"].cat.categories.get_loc("O_Create Offer") + 1)
    print("A_Submitted code: ", data["concept:name"].cat.categories.get_loc("A_Submitted") + 1)
    print("W_Validate application code: ", data["concept:name"].cat.categories.get_loc("W_Validate application") + 1)
    print("A_Complete code: ", data["concept:name"].cat.categories.get_loc("A_Complete") + 1)
    data["concept:name"] = data["concept:name"].cat.codes + 1
    vocab_sizes["concept:name"] = data["concept:name"].max()

    data["org:resource"] = data["org:resource"].fillna("Unknown")
    data["org:resource"] = pd.Categorical(data["org:resource"])
    data["org:resource"] = data["org:resource"].cat.codes + 1
    vocab_sizes["org:resource"] = data["org:resource"].max()

    data["Action"] = pd.Categorical(data["Action"])
    data["Action"] = data["Action"].cat.codes + 1
    vocab_sizes["Action"] = data["Action"].max()

    data["lifecycle:transition"] = pd.Categorical(data["lifecycle:transition"])
    data["lifecycle:transition"] = data["lifecycle:transition"].cat.codes + 1
    vocab_sizes["lifecycle:transition"] = data["lifecycle:transition"].max()

    data["case:LoanGoal"] = pd.Categorical(data["case:LoanGoal"])
    print("Case LoanGoal code: ", data["case:LoanGoal"].cat.categories.get_loc("Existing loan takeover") + 1)
    print("Case LoanGoal Car: ", data["case:LoanGoal"].cat.categories.get_loc("Car") + 1)
    data["case:LoanGoal"] = data["case:LoanGoal"].cat.codes + 1
    vocab_sizes["case:LoanGoal"] = data["case:LoanGoal"].max()

    data["case:ApplicationType"] = pd.Categorical(data["case:ApplicationType"])
    print("Case ApplicationType code: ", data["case:ApplicationType"].cat.categories.get_loc("New credit") + 1)
    data["case:ApplicationType"] = data["case:ApplicationType"].cat.codes + 1
    vocab_sizes["case:ApplicationType"] = data["case:ApplicationType"].max()

    data["NumberOfTerms"] = data["NumberOfTerms"].fillna(0)
    data["NumberOfTerms"] = pd.Categorical(data["NumberOfTerms"])
    data["NumberOfTerms"] = data["NumberOfTerms"].cat.codes + 1
    vocab_sizes["NumberOfTerms"] = data["NumberOfTerms"].max()

    scaler_ra = MinMaxScaler()
    data["case:RequestedAmount"] = data["case:RequestedAmount"].fillna(0)
    data["case:RequestedAmount"] = scaler_ra.fit_transform(data[["case:RequestedAmount"]])
    scalers["case:RequestedAmount"] = scaler_ra

    scaler_fw = MinMaxScaler()
    data["FirstWithdrawalAmount"] = data["FirstWithdrawalAmount"].fillna(0)
    data["FirstWithdrawalAmount"] = scaler_fw.fit_transform(data[["FirstWithdrawalAmount"]])
    scalers["FirstWithdrawalAmount"] = scaler_fw

    scaler_mc = MinMaxScaler()
    data["MonthlyCost"] = data["MonthlyCost"].fillna(0)
    data["MonthlyCost"] = scaler_mc.fit_transform(data[["MonthlyCost"]])
    scalers["MonthlyCost"] = scaler_mc

    scaler_cr = MinMaxScaler()
    data["CreditScore"] = data["CreditScore"].fillna(0)
    data["CreditScore"] = scaler_cr.fit_transform(data[["CreditScore"]])
    scalers["CreditScore"] = scaler_cr

    scaler_oa = MinMaxScaler()
    data["OfferedAmount"] = data["OfferedAmount"].fillna(0)
    data["OfferedAmount"] = scaler_oa.fit_transform(data[["OfferedAmount"]])
    scalers["OfferedAmount"] = scaler_oa

    data = data[['case:concept:name', 'label', 'concept:name', 'case:LoanGoal', 'CreditScore', 'Action', 'org:resource', 'lifecycle:transition', 'case:ApplicationType', 'case:RequestedAmount', 'FirstWithdrawalAmount', 'NumberOfTerms', 'MonthlyCost', 'OfferedAmount', 'time:timestamp', 'rule_1', 'rule_2', 'rule_3']]

    return create_ngrams(data, train_ids, test_ids), vocab_sizes, scalers