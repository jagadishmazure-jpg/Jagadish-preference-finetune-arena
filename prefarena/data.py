"""Load the committed prompts and measured outputs."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class DataError(ValueError):
    """Bad data file. Messages name the file and line, never the content."""


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise DataError(f"{path.name}:{n}: not valid JSON") from None
        if not isinstance(row, dict):
            raise DataError(f"{path.name}:{n}: expected an object")
        out.append(row)
    return out


@lru_cache(maxsize=1)
def prompts() -> list[dict]:
    return read_jsonl(DATA / "prompts.jsonl")


@lru_cache(maxsize=1)
def prompt_index() -> dict[str, dict]:
    return {p["id"]: p for p in prompts()}


@lru_cache(maxsize=None)
def rows(name: str) -> list[dict]:
    p = DATA / f"{name}.jsonl"
    return read_jsonl(p) if p.exists() else []
