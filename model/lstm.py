import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, vocab_sizes, config, num_classes, feature_names):
        super(LSTMModel, self).__init__()
        self.config = config
        torch.manual_seed(self.config)
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, 32, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        lstm_input_size = (32 * len(self.embeddings)) + (len(feature_names) - len(self.embeddings))
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate
        )
        self.fc = nn.Linear(config.hidden_size, self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = self.config.sequence_length
        embeddings_list = []
        numerical_features = []
        for name in self.embeddings.keys():
            index = self.feature_names.index(name)
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx].long()
            embeddings_list.append(self.embeddings[name](feature_data))
        if self.config.dataset == "sepsis":
            for name in ['InfectionSuspected', 'DiagnosticBlood', 'DisfuncOrg', 'SIRSCritTachypnea', 'Hypotensie', 'SIRSCritHeartRate', 'Infusion', 'DiagnosticArtAstrup', 'Age', 'DiagnosticIC', 'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther', 'SIRSCriteria2OrMore', 'DiagnosticXthorax', 'SIRSCritTemperature', 'DiagnosticUrinaryCulture', 'SIRSCritLeucos', 'Oligurie', 'DiagnosticLacticAcid', 'Hypoxie', 'DiagnosticUrinarySediment', 'DiagnosticECG', 'Leucocytes', 'CRP', 'LacticAcid']:
                index = self.feature_names.index(name)
                index = index * seq_len
                end_idx = index + seq_len
                feature_data = x[:, index:end_idx]
                numerical_features.append(feature_data)
        elif self.config.dataset == "bpi12":
            index = self.feature_names.index("case:AMOUNT_REQ")
            index = index * seq_len
            end_idx = index + seq_len
            feature_data = x[:, index:end_idx]
            numerical_features.append(feature_data)
        elif self.config.dataset == "traffic":
            for name in ["expense", "amount", "paymentAmount"]:
                index = self.feature_names.index(name)
                index = index * seq_len
                end_idx = index + seq_len
                feature_data = x[:, index:end_idx]
                numerical_features.append(feature_data)
        elif self.config.dataset == "bpi17":
            for name in ["CreditScore", "MonthlyCost", "OfferedAmount", "case:RequestedAmount", "FirstWithdrawalAmount"]:
                index = self.feature_names.index(name)
                index = index * seq_len
                end_idx = index + seq_len
                feature_data = x[:, index:end_idx]
                numerical_features.append(feature_data)

        numerical_features = torch.stack(numerical_features, dim=2)
        output = torch.cat(embeddings_list + [numerical_features], dim=2)
        return output
    
    def get_last_hidden(self, x):
        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)
        last_output = output[torch.arange(output.size(0)), lengths - 1]
        return last_output

    def forward(self, x):

        cat = self._get_embeddings(x)
        
        output, (hidden, _) = self.lstm(cat)
        last_hidden = hidden[-1]
        lengths = (x[:, :self.config.sequence_length] != 0).sum(1)
        last_output = output[torch.arange(output.size(0)), lengths - 1]

        out = self.fc(last_output)
        return self.sigmoid(out)