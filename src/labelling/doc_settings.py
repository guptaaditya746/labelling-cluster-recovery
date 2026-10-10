"""Settings of the evidence-document builder (``labelling evidence-docs``), read from the environment.

The names keep the ``HARNESS_DOC_`` prefix they had in the harness, so an existing ``.env`` or a
resumed build keeps working. Everything else the builder uses (model aliases, the LiteLLM endpoint,
the corpus, the graph settings such as ``HARNESS_SCHEMA_SCOPE``) comes from the harness settings, which
read the same ``.env`` and the profile named by ``HARNESS_PROFILE``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int, low: int, high: int | None = None) -> int:
    raw = os.environ.get(name, "").strip()
    value = int(raw) if raw else default
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name}={value} is outside {low}..{high if high is not None else '∞'}")
    return value


def _path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class DocSettings:
    # The question set: empty = questions/questions.json here; a path; or a file name under the
    # harness checkout's datasets/.
    question_set: str
    # Where documents are written; None = evidence_docs/ in this repository.
    root: Path | None
    # Papers screened per question: 0 = every indexed paper.
    max_papers: int
    # Passages of one paper the screening reader sees (highest-ranked for the question first; a paper
    # without a ranked passage is read from its first passages, so every paper is actually read).
    screen_passages: int
    # The question's scope in a relevant paper: the passages its reader cited (and their neighbours) plus
    # this many of its highest-ranked passages for the question by the harness ranker are extracted.
    # 0 = every passage of each relevant paper (the option for a final gold build when the model budget
    # allows: extraction calls grow from the cited + top-k passages of a paper to all of them, and a paper
    # can have a hundred; about 26,000 triples for nlp33, docs/EVIDENCE_DOCS.md). A lower value applied to an earlier, larger build cuts its triples to the new
    # scope without model calls; a higher value (or 0) extracts only the passages not extracted yet.
    extract_passages: int
    # Passages schema inference sees for one interpretation.
    schema_passages: int
    # Papers screened or extracted at the same time.
    concurrency: int
    # Model call timeout while building (no run deadline applies; building is offline).
    call_timeout_s: float
    # Papers marked not relevant drawn per question and interpretation for the screening check
    # (``labelling screening-sample``): people judge them, and the report estimates what screening missed.
    screening_sample: int = 5


def doc_settings() -> DocSettings:
    timeout = float(os.environ.get("HARNESS_DOC_CALL_TIMEOUT_S", "").strip() or 600)
    if timeout <= 0:
        raise ValueError("HARNESS_DOC_CALL_TIMEOUT_S must be positive")
    return DocSettings(
        question_set=os.environ.get("HARNESS_DOC_QUESTION_SET", "").strip(),
        root=_path("HARNESS_DOC_ROOT"),
        max_papers=_int("HARNESS_DOC_MAX_PAPERS", 0, 0),
        screen_passages=_int("HARNESS_DOC_SCREEN_PASSAGES", 10, 1),
        extract_passages=_int("HARNESS_DOC_EXTRACT_PASSAGES", 12, 0),
        schema_passages=_int("HARNESS_DOC_SCHEMA_PASSAGES", 40, 1),
        concurrency=_int("HARNESS_DOC_CONCURRENCY", 4, 1, 32),
        call_timeout_s=timeout,
        screening_sample=_int("HARNESS_DOC_SCREENING_SAMPLE", 5, 0),
    )
