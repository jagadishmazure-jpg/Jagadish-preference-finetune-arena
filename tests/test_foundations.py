"""Classic DL/NLP smoke: digits MLP/CNN, BiLSTM vs tiny DistilBERT, PII patterns, optional spaCy."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm-foundations"))

from llmfound.intent import examples, pretrained_name, run_intent  # noqa: E402
from llmfound.mlp_cnn import run_digits  # noqa: E402
from llmfound.pii import find_pii  # noqa: E402

NOTE = "Email mira@fenwold.example or call 415-555-0199 about ORD-104488 using card 4111 1111 1111 1111."


def test_mlp_and_cnn_beat_the_majority_class_on_digits():
    torch.manual_seed(123)
    result = run_digits(epochs=5, seed=0)
    torch.manual_seed(999)
    again = run_digits(epochs=1, seed=0)
    torch.manual_seed(1)
    same = run_digits(epochs=1, seed=0)
    assert result["mlp_beats_majority"] and result["cnn_beats_majority"]
    assert result["mlp"] > 0.5 and result["cnn"] > 0.5
    assert again["mlp"] == same["mlp"] and again["cnn"] == same["cnn"]


def test_bilstm_and_tiny_distilbert_beat_the_majority_intent():
    result = run_intent(epochs=8, seed=0)
    assert result["n_train"] > result["n_test"] > 0
    assert result["bilstm_beats_majority"] and result["distilbert_beats_majority"]
    assert pretrained_name() == "distilbert-base-uncased"
    labels = {label for _, label in examples()}
    assert labels == {"returns", "shipping", "warranty", "gift_card", "unsafe"}


def test_pii_patterns_find_structured_values_and_leave_policy_text_alone():
    labels = {hit["label"] for hit in find_pii(NOTE)}
    assert labels == {"EMAIL", "PHONE", "PAYMENT_CARD", "ORDER_ID"}
    assert find_pii("Unused items can be returned within 45 days with a receipt.") == []
    # A short policy number is not a payment card.
    assert find_pii("It takes 45 days.") == []


def test_spacy_component_emits_the_same_labels():
    spacy = pytest.importorskip("spacy")
    from llmfound.pii import build_spacy

    doc = build_spacy()(NOTE)
    assert {ent.label_ for ent in doc.ents} == {"EMAIL", "PHONE", "PAYMENT_CARD", "ORDER_ID"}
    assert spacy.blank("en").pipe_names == []
