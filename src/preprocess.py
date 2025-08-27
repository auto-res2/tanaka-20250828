import os
import numpy as np
import torch

# ------------------------------ Utilities ------------------------------

def set_seed(seed: int = 42):
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------ Synthetic Datasets ------------------------------

class SyntheticTextDataset(torch.utils.data.Dataset):
    """
    Three synthetic patterns to test robustness:
    - uniform: iid tokens
    - bigram: simple Markov chain with sparse transition structure
    - copyshift: token at t copies token at t-1 with prob p, otherwise random
    """
    def __init__(self, vocab_size=1000, seq_len=64, n_samples=1000, pattern="uniform", p_copy=0.7, seed=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.pattern = pattern
        self.p_copy = p_copy
        rng = np.random.default_rng(seed)

        if pattern == "uniform":
            self.data = rng.integers(low=0, high=vocab_size, size=(n_samples, seq_len), dtype=np.int64)
        elif pattern == "bigram":
            # Create sparse transition matrix with a few strong bigrams
            K = vocab_size
            trans = rng.random((K, K)) * 1e-3
            for _ in range(max(1, K // 50)):
                a = rng.integers(0, K)
                b = rng.integers(0, K)
                trans[a, b] += 1.0
            trans = trans / trans.sum(axis=1, keepdims=True)
            self.data = np.zeros((n_samples, seq_len), dtype=np.int64)
            self.data[:, 0] = rng.integers(0, K, size=(n_samples,), dtype=np.int64)
            for i in range(n_samples):
                for t in range(1, seq_len):
                    prev = self.data[i, t-1]
                    probs = trans[prev]
                    self.data[i, t] = rng.choice(K, p=probs)
        elif pattern == "copyshift":
            self.data = np.zeros((n_samples, seq_len), dtype=np.int64)
            self.data[:, 0] = rng.integers(0, vocab_size, size=(n_samples,), dtype=np.int64)
            for i in range(n_samples):
                for t in range(1, seq_len):
                    if rng.random() < p_copy:
                        self.data[i, t] = self.data[i, t-1]
                    else:
                        self.data[i, t] = rng.integers(0, vocab_size)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        return {"input_ids": x}


def build_dataloaders(vocab_size=1000, seq_len=64, n_train=800, n_val=200, pattern="uniform", batch_size=16, seed=0):
    train_ds = SyntheticTextDataset(vocab_size=vocab_size, seq_len=seq_len, n_samples=n_train, pattern=pattern, seed=seed)
    val_ds = SyntheticTextDataset(vocab_size=vocab_size, seq_len=seq_len, n_samples=n_val, pattern=pattern, seed=seed+1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
