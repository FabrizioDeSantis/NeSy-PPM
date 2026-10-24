import torch
import torch.nn as nn
<<<<<<< HEAD

from model.inputs import EventEncoder

=======
import torch.nn.functional as F
>>>>>>> 4f36cb968508f65a40d28fa775e60de82d73c536

class EventTransformer(nn.Module):
    """Transformer encoder over the events of a prefix, mean-pooled. Returns P(label = 1), shape (B, 1)."""

    def __init__(self, feature_names, vocab_sizes, seq_len, model_dim=128, num_layers=1, num_heads=2, dropout=0.1):
        super().__init__()
        self.encoder = EventEncoder(feature_names, vocab_sizes, seq_len)
        self.input_proj = nn.Linear(self.encoder.output_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.transformer(self.input_proj(self.encoder(x)) + self.positional_encoding)
        return self.sigmoid(self.classifier(h.mean(dim=1)))
