"""Byte-level byte-pair encoding, small enough to train on the Fenwold corpus in a test.

Ids 0-255 are raw bytes. Each merge takes the next id. pad, bos and eos are the three ids after the
merges, so they never appear inside a piece of text.
"""

from __future__ import annotations


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _unmerge(ids: list[int], token: int, pair: tuple[int, int]) -> list[int]:
    out = []
    for i in ids:
        if i == token:
            out.extend(pair)
        else:
            out.append(i)
    return out


class ByteBPE:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []
        self.pad_id = 256
        self.bos_id = 257
        self.eos_id = 258

    def train(self, text: str, num_merges: int) -> ByteBPE:
        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        for i in range(num_merges):
            counts: dict[tuple[int, int], int] = {}
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
            if not counts:
                break
            pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[pair] < 2:
                break
            ids = _merge(ids, pair, 256 + i)
            merges.append(pair)
        self.merges = merges
        base = 256 + len(merges)
        self.pad_id, self.bos_id, self.eos_id = base, base + 1, base + 2
        return self

    @property
    def vocab_size(self) -> int:
        return self.eos_id + 1

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        for i, pair in enumerate(self.merges):
            ids = _merge(ids, pair, 256 + i)
        return ids

    def decode(self, ids: list[int]) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        raw = [i for i in ids if i not in specials]
        for i, pair in reversed(list(enumerate(self.merges))):
            raw = _unmerge(raw, 256 + i, pair)
        return bytes(raw).decode("utf-8")
