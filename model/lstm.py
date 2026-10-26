import torch.nn as nn

from model.inputs import EventEncoder


class LSTMModel(nn.Module):
    """LSTM over the events of a prefix, classified from the last hidden state. Returns P(label = 1), shape (B, 1)."""

    def __init__(self, feature_names, vocab_sizes, seq_len, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = EventEncoder(feature_names, vocab_sizes, seq_len)
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
