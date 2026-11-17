"""MLP and a small CNN on the public digits set that ships with scikit-learn (8x8 images, 10 classes).

No download. A few epochs on CPU. The point is the two architectures and a majority-class baseline,
not a leaderboard number.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x.view(x.shape[0], -1))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        h = torch.relu(self.conv(x.view(x.shape[0], 1, 8, 8)))
        return self.fc(h.flatten(1))


def _split(x, y, seed: int, frac: float = 0.8):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    cut = int(frac * x.shape[0])
    return x[perm[:cut]], y[perm[:cut]], x[perm[cut:]], y[perm[cut:]]


def _fit(model: nn.Module, x, y, epochs: int, lr: float, seed: int) -> None:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def _accuracy(model, x, y) -> float:
    return float((model(x).argmax(-1) == y).float().mean())


def run_digits(epochs: int = 8, seed: int = 0) -> dict:
    from sklearn.datasets import load_digits

    bunch = load_digits()
    x = torch.tensor(bunch.data, dtype=torch.float32) / 16.0
    y = torch.tensor(bunch.target, dtype=torch.long)
    x_tr, y_tr, x_te, y_te = _split(x, y, seed)
    majority = int(torch.bincount(y_tr).argmax())
    baseline = float((y_te == majority).float().mean())
    torch.manual_seed(seed)
    mlp = SmallMLP()
    torch.manual_seed(seed + 1)
    cnn = SmallCNN()
    _fit(mlp, x_tr, y_tr, epochs, lr=1e-2, seed=seed)
    _fit(cnn, x_tr, y_tr, epochs, lr=1e-2, seed=seed)
    mlp_acc, cnn_acc = _accuracy(mlp, x_te, y_te), _accuracy(cnn, x_te, y_te)
    return {
        "majority": baseline,
        "mlp": mlp_acc,
        "cnn": cnn_acc,
        "mlp_beats_majority": mlp_acc > baseline,
        "cnn_beats_majority": cnn_acc > baseline,
    }
