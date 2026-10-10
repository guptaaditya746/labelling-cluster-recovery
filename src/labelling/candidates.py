"""Review candidates: the triples of the evidence documents (D), one record per triple and reading.

D is built by ``labelling evidence-docs``: every paper is screened, and each relevant paper's
question-scoped passages are extracted (the passages the reader cites plus the top-k by the harness
ranker; ``HARNESS_DOC_EXTRACT_PASSAGES=0`` extracts every passage). No system's runs are pooled, but D is
produced by the harness's own reader, ranker and extractor, so it can lean towards what those tools find
(docs/EVALUATION.md, section 8). Reviewers judge each triple against its passage; what they keep, with
its relevance, becomes the answer key. An ambiguous question's triples are reviewed per interpretation
(``reading``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .corpus import Corpus, norm
from .questions import Question

PRODUCER = "evidence-doc"


class Candidate(BaseModel):
    candidate_id: str
    question_id: str
    question: str
    s: str
    p: str
    o: str
    source: str                      # passage id (or fact/reference id)
    evidence: str                    # the text a reviewer judges against
    paper_id: str | None = None      # corpus key of the paper the evidence belongs to
    producers: list[str] = Field(default_factory=list)
    reading: str | None = None       # the D interpretation id ("i2"); None for a corpus-wide question
    reading_label: str | None = None
    reading_gloss: str | None = None
    triple_id: str | None = None     # the triple's id in D ("t00042")
    # Stratified review (labelling triage): "triaged" (the model put it in a reviewed tier) or "sample" (a
    # random draw from the rest); None when the question was not triaged. ``triage`` is the model's label.
    # Neither is shown to reviewers.
    stratum: str | None = None
    triage: str | None = None


def candidate_id(question_id: str, s: str, p: str, o: str, source: str, reading: str | None = None) -> str:
    key = "\x1f".join([question_id, norm(s), norm(p), norm(o), source] + ([reading] if reading else []))
    return "cand-" + hashlib.sha256(key.encode()).hexdigest()[:16]


def paper_identifier(corpus: Corpus, value: object) -> str:
    """A paper named by its corpus key ("p12", as D's extraction writes it) becomes the identifier the harness
    gives that paper in runs: its verified DBLP IRI, else the key. Anything else is unchanged.

    Without this, a gold triple about "p12" and the run's triple about the paper's DBLP IRI never share a
    subject: strict and entailed cannot match them and the soft judge is never asked."""
    text = str(value)
    key = text.strip()
    return corpus.paper_iri(key) if key in corpus.papers else text


def from_evidence_docs(corpus: Corpus, root: Path, questions: list[Question]) -> tuple[list[Candidate], list[str]]:
    """Candidates from ``<root>/<question id>/doc.json``; returns them and the questions without a document.

    Each triple is shown with the passage that states it (its first source); the passage text from the
    corpus, else the evidence span D recorded. Ambiguous questions keep their readings apart.
    """
    found: list[Candidate] = []
    missing: list[str] = []
    for question in questions:
        path = Path(root).expanduser() / question.id / "doc.json"
        if not path.exists():
            missing.append(question.id)
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        ambiguous = len(doc.get("interpretations") or []) > 1 or question.category == "ambiguous"
        for reading in doc.get("interpretations") or []:
            for triple in reading.get("triples") or []:
                found.append(_candidate(corpus, question, triple, reading if ambiguous else None))
    return found, missing


def _candidate(corpus: Corpus, question: Question, triple: dict[str, Any], reading: dict[str, Any] | None) -> Candidate:
    source = (triple.get("sources") or [{}])[0]
    source_id = str(source.get("source_id") or "")
    passage = corpus.passages.get(source_id)
    evidence = str(passage.get("text", "")) if passage else str(source.get("evidence") or "")
    paper = str(passage["paper_id"]) if passage else (str(triple.get("paper_id")) if triple.get("paper_id") else None)
    rid = str(reading["id"]) if reading else None
    return Candidate(
        # The id keeps D's own spelling, so records pushed before papers were canonicalised still match.
        candidate_id=candidate_id(question.id, triple["s"], triple["p"], triple["o"], source_id, rid),
        question_id=question.id, question=question.question,
        s=paper_identifier(corpus, triple["s"]), p=str(triple["p"]), o=paper_identifier(corpus, triple["o"]),
        source=source_id,
        evidence=evidence, paper_id=paper, producers=[PRODUCER],
        reading=rid, reading_label=str(reading.get("label") or "") if reading else None,
        reading_gloss=str(reading.get("gloss") or "") if reading else None,
        triple_id=str(triple.get("id")) if triple.get("id") else None,
    )


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            data = row.model_dump() if isinstance(row, BaseModel) else row
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_candidates(path: Path) -> list[Candidate]:
    return [Candidate.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]
