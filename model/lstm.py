import torch.nn as nn

from model.inputs import EventEncoder


class LSTMModel(nn.Module):
<<<<<<< HEAD
    """LSTM over the events of a prefix, classified from the last hidden state. Returns P(label = 1), shape (B, 1)."""

    def __init__(self, feature_names, vocab_sizes, seq_len, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = EventEncoder(feature_names, vocab_sizes, seq_len)
=======
    def __init__(self, vocab_sizes, config, num_classes, feature_names):
        super(LSTMModel, self).__init__()
        self.config = config
        torch.manual_seed(self.config.seed)
        self.feature_names = feature_names
        self.num_classes = num_classes
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(vocab_size + 1, 32, padding_idx=0)
            for feature, vocab_size in vocab_sizes.items()
        })
        lstm_input_size = (32 * len(self.embeddings)) + (len(feature_names) - len(self.embeddings))
>>>>>>> 4f36cb968508f65a40d28fa775e60de82d73c536
        self.lstm = nn.LSTM(
            input_size=self.encoder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.encoder(x))
        return self.sigmoid(self.fc(hidden[-1]))
