"""The screening check: how many relevant papers did D's screening mark "not relevant"?

D's coverage records a status for every paper (docs/EVIDENCE_DOCS.md). A paper marked ``not_relevant`` is
never extracted, so a wrong "not relevant" silently removes its facts from the gold. This module draws a
seeded random sample of those papers per question and interpretation (``labelling screening-sample``),
people judge each one in the ``<name>-screening`` Argilla dataset ("Is this paper relevant to the
question? If yes, which facts are missing?"), and ``labelling screening-report`` estimates the screening's
false-negative rate per section with a 95% Wilson interval, scaled to the number of papers marked not
relevant there. Facts people add on these records go into the gold like missing-facts lines
(``gold.missing_claims``).

A paper counts as relevant when any person who checked it says so (the conservative reading for a
completeness claim). Papers left unread (``no_text``, ``error``) are not in the sampling frame; the report
lists how many there were.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .corpus import Corpus
from .triage import sample_ids, wilson


def _sections(doc: dict[str, Any]) -> list[tuple[str | None, dict[str, Any]]]:
    sections = doc.get("interpretations") or []
    ambiguous = len(sections) > 1 or doc.get("question", {}).get("category") == "ambiguous"
    return [(str(reading["id"]) if ambiguous else None, reading) for reading in sections]


def record_id(question_id: str, reading: str | None, paper: str) -> str:
    return "screening-" + "-".join(part for part in (question_id, reading, paper) if part)


def paper_markdown(corpus: Corpus, paper: str) -> str:
    """The paper as a reviewer reads it: title, key and every passage in corpus order."""
    passages = [row for row in corpus.passages.values() if str(row.get("paper_id")) == paper]
    lines = [f"**{corpus.title(paper)}** (`{paper}`)", ""]
    lines += [f"`{row['passage_id']}` {row.get('text', '')}" for row in passages]
    if not passages:
        lines.append("_No passage of this paper is in the corpus._")
    return "\n\n".join(lines)


def sample_records(corpus: Corpus, root: Path, question_ids: list[str] | None, *, size: int, seed: str,
                   ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Per question and interpretation, ``size`` papers drawn at random (fixed by ``seed``) from those its
    coverage marks ``not_relevant``. Returns the Argilla records and the strata (``<question>`` or
    ``<question>/<reading>`` -> sizes and drawn paper ids)."""
    records: list[dict[str, Any]] = []
    strata: dict[str, dict[str, Any]] = {}
    found = sorted(path.parent.name for path in Path(root).expanduser().glob("*/doc.json"))
    for question_id in found:
        if question_ids is not None and question_id not in question_ids:
            continue
        doc = json.loads((Path(root).expanduser() / question_id / "doc.json").read_text(encoding="utf-8"))
        question = str(doc.get("question", {}).get("question", ""))
        for reading, section in _sections(doc):
            coverage = section.get("coverage") or []
            frame = [str(row["paper_id"]) for row in coverage if row.get("status") == "not_relevant"]
            drawn = sample_ids(frame, size, f"{seed}:{question_id}:{reading or ''}")
            chosen = [paper for paper in frame if paper in drawn]
            key = f"{question_id}/{reading}" if reading else question_id
            strata[key] = {"question_id": question_id, "reading": reading, "not_relevant": len(frame),
                           "unread": sum(1 for row in coverage if row.get("status") in {"no_text", "error"}),
                           "relevant": sum(1 for row in coverage if row.get("status") == "relevant"),
                           "sampled": len(chosen), "papers": chosen, "seed": seed}
            text = (f"{question}\n\nInterpretation: {section.get('label')} — {section.get('gloss')}"
                    if reading else question)
            records += [{"id": record_id(question_id, reading, paper), "question": text,
                         "paper": paper_markdown(corpus, paper), "question_id": question_id,
                         "reading": reading or "", "paper_id": paper, "not_relevant_total": len(frame),
                         "sample_size": len(chosen)} for paper in chosen]
    return records, strata


def _verdicts(record: dict[str, Any]) -> list[str]:
    return [str(response.get("value")) for response in record.get("responses", [])
            if response.get("status") == "submitted" and response.get("question") == "relevant"]


def report(reviews: dict[str, list[dict[str, Any]]], strata: dict[str, dict[str, Any]] | None = None,
           ) -> dict[str, Any]:
    """Per section: papers checked, found relevant, the false-negative rate with a 95% Wilson interval, and
    the estimated relevant papers among all marked not relevant (rate x N, interval x N).

    N comes from the records' metadata (``not_relevant_total``), else from ``strata``. ``overall`` adds the
    per-section estimates up (a stratified total); its ``checked_rate_95`` is the plain Wilson interval of
    the pooled checked papers, which weights sections by their sample, not by N.
    """
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"checked": 0, "relevant": 0, "N": None,
                                                             "papers_relevant": []})
    for record in reviews.get("screening", []):
        metadata = record.get("metadata", {})
        question_id = str(metadata.get("question_id") or "")
        reading = str(metadata.get("reading") or "") or None
        key = f"{question_id}/{reading}" if reading else question_id
        group = groups[key]
        if metadata.get("not_relevant_total") is not None:
            group["N"] = int(metadata["not_relevant_total"])
        verdicts = _verdicts(record)
        if not verdicts:
            continue
        group["checked"] += 1
        if "yes" in verdicts:
            group["relevant"] += 1
            group["papers_relevant"].append(str(metadata.get("paper_id") or record.get("id")))
    for key, sizes in (strata or {}).items():
        if groups[key]["N"] is None:
            groups[key]["N"] = int(sizes.get("not_relevant", 0))
    sections: dict[str, Any] = {}
    total_n, total_missed, low_sum, high_sum, checked, relevant = 0, 0.0, 0.0, 0.0, 0, 0
    for key in sorted(groups):
        group = groups[key]
        n_total = group["N"] or 0
        k, n = group["relevant"], group["checked"]
        low, high = wilson(k, n)
        rate = k / n if n else None
        sections[key] = {
            "not_relevant": n_total, "checked": n, "found_relevant": k,
            "false_negative_rate": round(rate, 4) if rate is not None else None,
            "false_negative_rate_95": [round(low, 4), round(high, 4)],
            "estimated_missed_papers": round(n_total * rate, 1) if rate is not None else None,
            "estimated_missed_papers_95": [round(n_total * low, 1), round(n_total * high, 1)],
            "papers_found_relevant": sorted(group["papers_relevant"]),
        }
        total_n += n_total
        checked += n
        relevant += k
        if rate is not None:
            total_missed += n_total * rate
        low_sum += n_total * low
        high_sum += n_total * high
    pooled_low, pooled_high = wilson(relevant, checked)
    return {
        "sections": sections,
        "overall": {
            "not_relevant": total_n, "checked": checked, "found_relevant": relevant,
            "estimated_missed_papers": round(total_missed, 1),
            # Summing per-section bounds is conservative (wider than a joint 95% interval).
            "estimated_missed_papers_bounds": [round(low_sum, 1), round(high_sum, 1)],
            "checked_rate": round(relevant / checked, 4) if checked else None,
            "checked_rate_95": [round(pooled_low, 4), round(pooled_high, 4)],
        },
    }
