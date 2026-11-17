"""PII spans for shop text: email, phone, payment-card number, order id.

The detector is a set of patterns. ``build_spacy`` wraps the same patterns as a spaCy pipeline
component that writes ``doc.ents``. spaCy is optional; CI runs the pattern detector either way.
A statistical PERSON tagger is only used when the caller passes an already-loaded model.
"""

from __future__ import annotations

import re
import threading

PATTERNS = (
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("PAYMENT_CARD", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")),
    ("ORDER_ID", re.compile(r"\bORD-\d{6,}\b")),
)


def find_pii(text: str) -> list[dict]:
    hits = []
    for label, pattern in PATTERNS:
        for match in pattern.finditer(text):
            hits.append({"label": label, "start": match.start(), "end": match.end(), "text": match.group()})
    hits.sort(key=lambda h: (h["start"], h["label"]))
    return hits


class _PiiComponent:
    def __init__(self, _nlp):
        self._nlp = _nlp

    def __call__(self, doc):
        import spacy

        spans = []
        for hit in find_pii(doc.text):
            span = doc.char_span(hit["start"], hit["end"], label=hit["label"], alignment_mode="expand")
            if span is not None:
                spans.append(span)
        doc.ents = spacy.util.filter_spans(spans)
        return doc


_FACTORY_LOCK = threading.Lock()


def build_spacy(nlp=None):
    """Blank English pipeline plus the PII component. Imports spaCy."""
    import spacy

    with _FACTORY_LOCK:
        if not spacy.Language.has_factory("fenwold_pii"):
            @spacy.Language.factory("fenwold_pii")
            def _factory(nlp, name):
                return _PiiComponent(nlp)

    nlp = nlp or spacy.blank("en")
    if "fenwold_pii" not in nlp.pipe_names:
        nlp.add_pipe("fenwold_pii", last=True)
    return nlp


def spacy_labels(text: str) -> list[str]:
    doc = build_spacy()(text)
    return [ent.label_ for ent in doc.ents]
