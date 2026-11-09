"""The synthetic prompt set: deterministic, split by template, and covering every policy and risk."""

import subprocess
import sys
from collections import defaultdict

from conftest import ROOT

from prefarena.company import POLICIES, context
from prefarena.prompts import UNSAFE, build_prompts


def test_generator_is_deterministic_and_matches_committed_file():
    assert build_prompts() == build_prompts()
    out = subprocess.run([sys.executable, "scripts/build_dataset.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_ids_and_texts_are_unique(prompts):
    assert len({p["id"] for p in prompts}) == len(prompts) == len({p["prompt"] for p in prompts})


def test_a_template_never_spans_two_splits(prompts):
    splits = defaultdict(set)
    for p in prompts:
        splits[p["template"]].add(p["split"])
    assert all(len(s) == 1 for s in splits.values())


def test_every_policy_and_every_risk_kind_appears(prompts):
    assert {p["policy"] for p in prompts if p["kind"] == "policy"} == set(POLICIES)
    assert {p["topic"] for p in prompts if p["kind"] == "unsafe"} == set(UNSAFE)
    test = [p for p in prompts if p["split"] == "test"]
    assert sum(p["kind"] == "unsafe" for p in test) >= 20


def test_context_gives_the_policy_sentence_with_its_id():
    assert context("SHP-1") == "[SHP-1] Standard shipping is free on orders over $60."
    assert context(None) == "No policy covers this request."


def test_each_policy_key_value_is_in_its_sentence():
    for pid, (_, sentence, keys) in POLICIES.items():
        assert keys[0].lower() in sentence.lower(), pid
