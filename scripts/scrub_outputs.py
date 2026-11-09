"""Scrub real company names out of model outputs, in place (idempotent).

Small open models sometimes name real stores, carriers or payment brands on their own. Every response
in data/candidates.jsonl and data/generations.jsonl is passed through check_brands.scrub, which swaps
each name for a fixed fictional one with the same number of words. gen_candidates.py and tune.py scrub
as they write, so this is only needed for files written before that. Counts go to data/scrub_log.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_brands import scrub  # noqa: E402


def main() -> int:
    log = {}
    for name in ("candidates", "generations"):
        path = ROOT / "data" / f"{name}.jsonl"
        if not path.exists():
            continue
        rows, n_rows, n_names = [], 0, 0
        for line in path.read_text().splitlines():
            r = json.loads(line)
            r["response"], n = scrub(r["response"])
            n_rows += n > 0
            n_names += n
            rows.append(json.dumps(r))
        path.write_text("\n".join(rows) + "\n")
        log[name] = {"responses_changed": n_rows, "names_replaced": n_names}
    log_path = ROOT / "data" / "scrub_log.json"
    old = json.loads(log_path.read_text()) if log_path.exists() else {}
    for k, v in log.items():  # keep the first (largest) counts; later runs find nothing new
        if v["names_replaced"] or k not in old:
            old[k] = v
    log_path.write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")
    print(json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
