import numpy as np
import torch
from torch.utils.data import Dataset


class NeSyDataset(Dataset):
    """
    Prefixes built by data/common.create_ngrams, as a (N, F * L) float tensor: the F features of
    each prefix, padded to L events, stored one after the other (feature-major).
    """

    def __init__(self, sequences, labels):
        self.sequences = torch.as_tensor(np.asarray(sequences, dtype=np.float32)).reshape(len(labels), -1)
        self.labels = torch.as_tensor(labels, dtype=torch.float)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
