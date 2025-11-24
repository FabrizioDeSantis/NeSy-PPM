import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class EventTransformer(nn.Module):
    def __init__(self, vocab_sizes, config, feature_names, model_dim, num_classes, max_len, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.config = config
        self.feature_names = feature_names
        self.numerical_features = [name for name in feature_names if name not in vocab_sizes.keys()]
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, 32, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        transformer_input_size = (32 * len(self.embeddings)) + len(self.numerical_features)
        self.input_proj = nn.Linear(transformer_input_size, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, model_dim))
        torch.manual_seed(self.config.seed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        classifier_input_size = model_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, classifier_input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size // 2, num_classes)
        )

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
    
    def forward(self, x, mask=None):

        x = self._get_embeddings(x)
        x = self.input_proj(x)
        x = x + self.positional_encoding[:, :x.size(1), :]

        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        pooled = x.mean(dim=1)

        output = self.classifier(pooled)

        return self.sigmoid(output)