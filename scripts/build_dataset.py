"""Write data/prompts.jsonl from the generator. Deterministic; --check fails if the file differs (CI)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prefarena.prompts import build_prompts  # noqa: E402


def main() -> int:
    text = "".join(json.dumps(p, sort_keys=True) + "\n" for p in build_prompts())
    path = ROOT / "data" / "prompts.jsonl"
    if "--check" in sys.argv:
        ok = path.exists() and path.read_text(encoding="utf-8") == text
        print("prompts", "match" if ok else "out of date")
        return 0 if ok else 1
    path.write_text(text, encoding="utf-8")
    print("prompts written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
