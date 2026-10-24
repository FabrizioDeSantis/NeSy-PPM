import torch
import torch.nn as nn


class EventEncoder(nn.Module):
    """
    Turns the flat model input into one vector per event.

    The input x has shape (B, F * L): F features, each padded to L events and stored one after the other
    (feature-major, see data/common.create_ngrams). Features with a vocabulary (vocab_sizes) are embedded,
    the others (numerical features, and the Declare truth values appended by nesy/experiments.ModelInput)
    are used as they are. forward(x) returns a (B, L, D) tensor.
    """

    def __init__(self, feature_names, vocab_sizes, seq_len, embedding_dim=32):
        super().__init__()
        self.seq_len = seq_len
        offset = {name: i * seq_len for i, name in enumerate(feature_names)}
        self.categorical = [(name, offset[name]) for name in feature_names if name in vocab_sizes]
        self.numerical = [offset[name] for name in feature_names if name not in vocab_sizes]
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(int(vocab_sizes[name]) + 1, embedding_dim, padding_idx=0)
            for name, _ in self.categorical
        })
        self.output_dim = embedding_dim * len(self.categorical) + len(self.numerical)

    def _feature(self, x, offset):
        return x[:, offset:offset + self.seq_len]

    def forward(self, x):
        parts = [self.embeddings[name](self._feature(x, offset).long()) for name, offset in self.categorical]
        if self.numerical:
            parts.append(torch.stack([self._feature(x, offset) for offset in self.numerical], dim=2))
        return torch.cat(parts, dim=2)
