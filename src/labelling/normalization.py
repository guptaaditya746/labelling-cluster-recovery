"""Shared canonicalization rules for corpus_10 gold export and harness scoring."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(value: object) -> str:
    """Apply only the agreed baseline normalization: Unicode, case and whitespace."""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def load_vocabulary(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def question_id_for_text(question: str, vocabulary: dict[str, Any]) -> str | None:
    wanted = normalize_text(question)
    for question_id, text in vocabulary.get("question_texts", {}).items():
        if normalize_text(text) == wanted:
            return question_id
    return None


def _aliases(canonical_to_aliases: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for canonical, aliases in canonical_to_aliases.items():
        result[normalize_text(canonical)] = canonical
        if isinstance(aliases, list):
            for alias in aliases:
                result[normalize_text(alias)] = canonical
    return result


def _canonical(value: object, canonical_to_aliases: dict[str, Any]) -> str:
    raw = normalize_text(value)
    return _aliases(canonical_to_aliases).get(raw, raw)


def _question_vocab(question_id: str | None, vocabulary: dict[str, Any]) -> dict[str, Any]:
    return vocabulary.get("questions", {}).get(question_id or "", {})


def _paper_id(subject: object, vocabulary: dict[str, Any]) -> str:
    raw = normalize_text(subject)
    for paper_id, aliases in vocabulary.get("paper_aliases", {}).items():
        if raw == normalize_text(paper_id) or raw in {normalize_text(alias) for alias in aliases}:
            return paper_id
    return raw


def canonicalize_triple(
    triple: dict[str, Any], question_id: str | None, vocabulary: dict[str, Any]
) -> dict[str, Any]:
    """Canonicalize subject, relation and object using one question's vocabulary."""
    question = _question_vocab(question_id, vocabulary)
    relation = _canonical(triple.get("p", ""), question.get("relations", {}))
    if question.get("subject_type") == "theme_from_q01":
        theme_vocab = _question_vocab("q01", vocabulary)
        subject = _canonical(triple.get("s", ""), theme_vocab.get("aliases", {}).get("theme", {}))
        subject = _canonical(subject, {label: [] for label in theme_vocab.get("canonical_labels", {}).get("theme", [])})
    else:
        subject = _paper_id(triple.get("s", ""), vocabulary)
    object_type = question.get("object_type", {}).get(relation)
    object_ = str(triple.get("o", "") or "")
    if object_type:
        object_ = _canonical(object_, question.get("aliases", {}).get(object_type, {}))
        object_ = _canonical(
            object_, {label: [] for label in question.get("canonical_labels", {}).get(object_type, [])}
        )
    return {**triple, "s": subject, "p": relation, "o": object_}


def canonical_triple_key(
    triple: dict[str, Any], question_id: str | None, vocabulary: dict[str, Any]
) -> tuple[str, str, str]:
    row = canonicalize_triple(triple, question_id, vocabulary)
    return tuple(normalize_text(row.get(key, "")) for key in ("s", "p", "o"))
