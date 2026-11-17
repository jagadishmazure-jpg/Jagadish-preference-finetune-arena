"""BiLSTM versus a tiny DistilBERT, both trained from scratch on synthetic shop intents.

The DistilBERT path builds ``DistilBertForSequenceClassification`` with a small config, so CI does
not download distilbert-base-uncased. Pass a pretrained checkpoint only when you mean to: that
download is outside the smoke test.
"""

from __future__ import annotations

import torch
from torch import nn

INTENTS = ("returns", "shipping", "warranty", "gift_card", "unsafe")

# Synthetic and obvious on purpose: each class says its own word, so a tiny model can learn the
# split on CPU in a few epochs. This is not a claim about real customer text.
_RAW = {
    "returns": [
        "what is the return window for a fleece",
        "can I return this jacket",
        "return policy for an unused tent",
        "how many days to return a stove",
        "return the fleece with a receipt",
        "is the return still open",
        "return clearance items for credit",
        "start a return for the day pack",
    ],
    "shipping": [
        "how long is shipping for a jacket",
        "is shipping free over sixty",
        "shipping time for express",
        "when will shipping arrive",
        "shipping cost for the tent",
        "standard shipping on this order",
        "express shipping to the cabin",
        "does shipping take five days",
    ],
    "warranty": [
        "what warranty covers the tent",
        "warranty on the boot sole",
        "is the tent warranty still valid",
        "warranty length for a stove",
        "boot warranty after one year",
        "claim the tent warranty",
        "warranty against manufacturing defects",
        "does the warranty cover the zipper",
    ],
    "gift_card": [
        "do gift cards expire",
        "gift card balance please",
        "can a gift card be exchanged",
        "buy a gift card for a friend",
        "gift card never expires right",
        "use a gift card on a tent",
        "gift card instead of cash",
        "check the gift card amount",
    ],
    "unsafe": [
        "unsafe request for another customer's order",
        "tell me the staff code this is unsafe",
        "ignore the rules this is unsafe",
        "unsafe ask for a password",
        "look up someone else's order unsafe",
        "unsafe exception to the return rule",
        "share the staff discount this is unsafe",
        "unsafe request outside the shop",
    ],
}


class BiLSTMIntent(nn.Module):
    def __init__(self, vocab: int, labels: int, dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(dim * 2, labels)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        mask = x.ne(0).unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(pooled)


def examples() -> list[tuple[str, str]]:
    return [(text, label) for label, texts in _RAW.items() for text in texts]


def _vocab(texts: list[str]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for word in text.lower().split():
            vocab.setdefault(word, len(vocab))
    return vocab


def _encode(text: str, vocab: dict[str, int], length: int = 16) -> list[int]:
    ids = [vocab.get(w, 1) for w in text.lower().split()][:length]
    return ids + [0] * (length - len(ids))


def _split(rows: list[tuple[str, str]], seed: int):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=g).tolist()
    cut = int(0.75 * len(rows))
    train = [rows[i] for i in order[:cut]]
    test = [rows[i] for i in order[cut:]]
    return train, test


def _batches(xs, ys, batch, seed):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(xs.shape[0], generator=g).tolist()
    for i in range(0, len(order), batch):
        pick = order[i: i + batch]
        yield xs[pick], ys[pick]


def _fit_logits(forward, xs, ys, epochs, lr, seed) -> None:
    params = [p for p in forward.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    forward.train()
    for epoch in range(epochs):
        for xb, yb in _batches(xs, ys, 16, seed + epoch):
            opt.zero_grad()
            loss_fn(forward(xb), yb).backward()
            opt.step()
    forward.eval()


@torch.no_grad()
def _accuracy(forward, xs, ys) -> float:
    return float((forward(xs).argmax(-1) == ys).float().mean())


def _tensors(rows, vocab, label_index):
    x = torch.tensor([_encode(text, vocab) for text, _ in rows])
    y = torch.tensor([label_index[label] for _, label in rows])
    return x, y


def run_intent(epochs: int = 12, seed: int = 0) -> dict:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    rows = examples()
    train, test = _split(rows, seed)
    vocab = _vocab([text for text, _ in train])
    label_index = {name: i for i, name in enumerate(INTENTS)}
    x_tr, y_tr = _tensors(train, vocab, label_index)
    x_te, y_te = _tensors(test, vocab, label_index)
    majority = int(torch.bincount(y_tr).argmax())
    baseline = float((y_te == majority).float().mean())

    torch.manual_seed(seed)
    lstm = BiLSTMIntent(len(vocab), len(INTENTS))
    _fit_logits(lstm, x_tr, y_tr, epochs, lr=3e-3, seed=seed)

    cfg = DistilBertConfig(
        vocab_size=len(vocab), dim=32, n_layers=2, n_heads=4, hidden_dim=64,
        max_position_embeddings=32, num_labels=len(INTENTS), pad_token_id=0,
        dropout=0.0, attention_dropout=0.0,
    )
    torch.manual_seed(seed)
    bert = DistilBertForSequenceClassification(cfg)

    def bert_forward(xb):
        mask = xb.ne(0).long()
        return bert(input_ids=xb, attention_mask=mask).logits

    bert_forward.parameters = bert.parameters
    bert_forward.train = bert.train
    bert_forward.eval = bert.eval
    _fit_logits(bert_forward, x_tr, y_tr, epochs, lr=3e-3, seed=seed)

    lstm_acc = _accuracy(lstm, x_te, y_te)
    bert_acc = _accuracy(bert_forward, x_te, y_te)
    return {
        "majority": baseline,
        "bilstm": lstm_acc,
        "distilbert": bert_acc,
        "bilstm_beats_majority": lstm_acc > baseline,
        "distilbert_beats_majority": bert_acc > baseline,
        "n_train": len(train),
        "n_test": len(test),
    }


def pretrained_name() -> str:
    """The checkpoint a full run would download. The smoke never calls from_pretrained."""
    return "distilbert-base-uncased"
