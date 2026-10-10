"""Argilla datasets for reviewing the evidence documents (D), and reading the reviews back.

Three datasets per gold version, plus two checks:

- ``<name>-review``: the review records of D, one reviewer each (you). Review is stratified
  (``labelling triage`` + ``labelling pool``): every triple triage put in a reviewed tier plus a random
  sample of the rest, not every triple of D (``--all`` reviews every triple).
- ``<name>-overlap``: the same, for the agreement subset (about 20% of the
  questions), two reviewers each (you and your professor), done
  independently; Cohen's kappa is computed on it.
- ``<name>-missing``: one record per question (per interpretation of an ambiguous
  question) to add correct facts the document missed, so recall is not bounded by D.
- ``<name>-screening``: a random sample of the papers D's screening marked not relevant
  (``labelling screening-sample``); people say whether each is relevant after all and add its facts,
  and ``labelling screening-report`` estimates the screening's false-negative rate.
- ``<name>-judge``: a sample of the harness judge's "same fact?" decisions,
  answered by people without seeing the judge's answer; kappa between the two
  decides whether the soft level may use the judge (docs/EVALUATION.md).

Record ids are candidate ids: pushing again updates records instead of
duplicating them.
"""

from __future__ import annotations

import os
from typing import Any

from .candidates import Candidate

VERDICT = {
    "correct": "Correct, and the evidence supports it",
    "partial": "Partly correct: write the fixed claim below",
    "wrong": "Wrong, or the evidence does not support it",
}
RELEVANCE = {
    "answers": "Answers the question",
    "supports": "Needed context for the answer (metric, benchmark, setting, comparator)",
    "irrelevant": "Not relevant to the question",
}
WEIGHT = {
    "vital": "Vital: an answer without this fact misses the point of the question",
    "okay": "Okay: worth having, but a good answer could leave it out",
}
SAME_FACT = {"yes": "Yes, the candidate states the reference fact", "no": "No"}
CORRECTION_FORMAT = "subject | relation | object"
MISSING_FORMAT = f"{CORRECTION_FORMAT} | passage_id | vital or okay"
GUIDELINES = (
    "Judge each candidate claim against its evidence passage only. See guidelines/annotation.md. "
    f"If a claim is partly correct, write the fixed claim as `{CORRECTION_FORMAT}`. For a kept claim, "
    "mark it vital or okay and list other acceptable wordings of the object (or whole claims), one per line."
)


def client() -> Any:
    import argilla as rg
    import httpx

    url = os.getenv("ARGILLA_API_URL", "http://127.0.0.1:36900")
    try:
        return rg.Argilla(api_url=url, api_key=os.getenv("ARGILLA_API_KEY"))
    except httpx.ConnectError as exc:
        raise SystemExit(
            f"Argilla is not reachable at {url} ({exc}). Argilla runs on Omen: check its stack there "
            "(`docker compose ps` in ~/stacks/argilla), then open the 'argilla' tunnel, laptop side first "
            "(`omen-tunnels laptop start argilla`), then on this node (`omen-tunnels compute start argilla`). "
            "See infra/README.md."
        ) from None


def review_settings(min_submitted: int) -> Any:
    import argilla as rg

    return rg.Settings(
        guidelines=GUIDELINES,
        fields=[
            rg.TextField(name="question", title="Question"),
            rg.TextField(name="claim", title="Candidate claim", use_markdown=True),
            rg.TextField(name="evidence", title="Evidence passage", use_markdown=True),
        ],
        questions=[
            rg.LabelQuestion(name="verdict", title="Is the claim correct?", labels=VERDICT),
            rg.LabelQuestion(name="relevance", title="Is it relevant to the question?", labels=RELEVANCE),
            rg.LabelQuestion(name="weight", title="If kept: how important is it?", labels=WEIGHT,
                             required=False),
            rg.TextQuestion(name="variants", title="Other acceptable wordings, one per line "
                            f"(object text, or {CORRECTION_FORMAT})", required=False),
            rg.TextQuestion(name="correction", title=f"Fixed claim ({CORRECTION_FORMAT})", required=False),
            rg.TextQuestion(name="note", title="Note (optional)", required=False),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="question_id", title="Question"),
            rg.TermsMetadataProperty(name="reading", title="Interpretation"),
            rg.TermsMetadataProperty(name="paper", title="Paper"),
            rg.TermsMetadataProperty(name="producers", visible_for_annotators=False),
        ],
        distribution=rg.TaskDistribution(min_submitted=min_submitted),
    )


def missing_settings(min_submitted: int = 1) -> Any:
    import argilla as rg

    return rg.Settings(
        guidelines=(
            "Add correct, question-relevant claims the evidence document does not contain. "
            f"One per line: `{MISSING_FORMAT}` (the weight is optional, default okay). "
            "Leave empty if nothing is missing."
        ),
        fields=[
            rg.TextField(name="question", title="Question"),
            rg.TextField(name="scope", title="Papers and passages in scope", use_markdown=True),
        ],
        questions=[
            rg.TextQuestion(name="missing", title="Missing claims", required=False),
            rg.LabelQuestion(
                name="done",
                title="Reviewed",
                labels={
                    "yes": "I checked this question",
                    "no": "I have not checked this question",
                },
            ),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="question_id", title="Question"),
            rg.TermsMetadataProperty(name="reading", title="Interpretation"),
        ],
        distribution=rg.TaskDistribution(min_submitted=min_submitted),
    )

def judge_settings(min_submitted: int = 1) -> Any:
    import argilla as rg

    return rg.Settings(
        guidelines=("Does the candidate state the same fact as the reference? The subjects already refer to "
                    "the same entity. Answer yes only if a reader of the candidate learns the reference fact; "
                    "a vaguer, more general or different claim is no. See docs/EVALUATION.md."),
        fields=[rg.TextField(name="reference", title="Reference (gold) fact", use_markdown=True),
                rg.TextField(name="candidate", title="Candidate (run) fact", use_markdown=True)],
        questions=[rg.LabelQuestion(name="same_fact", title="Same fact?", labels=SAME_FACT)],
        metadata=[rg.TermsMetadataProperty(name="judge", visible_for_annotators=False)],
        distribution=rg.TaskDistribution(min_submitted=min_submitted),
    )


SCREENING_RELEVANT = {
    "yes": "Relevant: it holds facts that answer the question (or interpretation) or are needed to read the answer",
    "no": "Not relevant",
}


def screening_settings(min_submitted: int = 1) -> Any:
    import argilla as rg

    return rg.Settings(
        guidelines=(
            "The evidence document's screening marked this paper not relevant to the question. Is this paper "
            "relevant to the question? If yes, which facts are missing? Add each as "
            f"`{MISSING_FORMAT}`, one per line (the weight is optional, default okay). "
            "See guidelines/annotation.md, section 5."
        ),
        fields=[
            rg.TextField(name="question", title="Question"),
            rg.TextField(name="paper", title="Paper (every passage)", use_markdown=True),
        ],
        questions=[
            rg.LabelQuestion(name="relevant", title="Is this paper relevant to the question?",
                             labels=SCREENING_RELEVANT),
            rg.TextQuestion(name="missing", title=f"If yes: the missing facts ({MISSING_FORMAT})", required=False),
            rg.TextQuestion(name="note", title="Note (optional)", required=False),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="question_id", title="Question"),
            rg.TermsMetadataProperty(name="reading", title="Interpretation"),
            rg.TermsMetadataProperty(name="paper_id", title="Paper"),
            # The stratum: papers marked not relevant in this section, and how many were drawn.
            rg.IntegerMetadataProperty(name="not_relevant_total", visible_for_annotators=False),
            rg.IntegerMetadataProperty(name="sample_size", visible_for_annotators=False),
        ],
        distribution=rg.TaskDistribution(min_submitted=min_submitted),
    )


def push_screening(rows: list[dict[str, Any]], *, name: str, workspace: str) -> int:
    """Rows from ``labelling screening-sample`` (``screening.sample_records``)."""
    api = client()
    dataset = ensure_dataset(api, f"{name}-screening", workspace, screening_settings())
    if rows:
        dataset.records.log(rows)
    return len(rows)


def _fact_markdown(fact: dict[str, str]) -> str:
    return f"**{fact.get('subject', '')}** — _{fact.get('relation', '')}_ — **{fact.get('object', '')}**"


def judge_record_id(row: dict[str, Any]) -> str:
    return str(row["key"])[:32]


def push_judge(rows: list[dict[str, Any]], *, name: str, workspace: str) -> int:
    """Rows from the harness judge cache; the judge's answer is stored hidden from annotators."""
    api = client()
    dataset = ensure_dataset(api, f"{name}-judge", workspace, judge_settings())
    dataset.records.log([
        {"id": judge_record_id(row), "reference": _fact_markdown(row["gold"]), "candidate": _fact_markdown(row["run"]),
         "judge": "yes" if row["same_fact"] else "no"}
        for row in rows
    ])
    return len(rows)


def claim_markdown(item: Candidate) -> str:
    return f"**{item.s}** — _{item.p}_ — **{item.o}**"


def evidence_markdown(item: Candidate, title: str) -> str:
    return f"`{item.source}` · {title}\n\n> {item.evidence}"


def question_text(question: str, label: str | None, gloss: str | None) -> str:
    """The question, and for an ambiguous one the interpretation the triple is judged under."""
    return f"{question}\n\nInterpretation: {label} — {gloss}" if label else question


def review_records(items: list[Candidate], titles: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.candidate_id,
            "question": question_text(item.question, item.reading_label, item.reading_gloss),
            "claim": claim_markdown(item),
            "evidence": evidence_markdown(item, titles.get(item.paper_id or "", "")),
            "question_id": item.question_id,
            "reading": item.reading or "",
            "paper": item.paper_id or "",
            # Hidden metadata: the stratum (to follow progress), never the triage label.
            "producers": ",".join(item.producers + ([f"stratum:{item.stratum}"] if item.stratum else [])),
        }
        for item in items
    ]


def ensure_dataset(api: Any, name: str, workspace: str, settings: Any) -> Any:
    import argilla as rg

    existing = api.datasets(name=name, workspace=workspace)
    if existing is not None:
        return existing
    return rg.Dataset(name=name, workspace=workspace, settings=settings, client=api).create()


def push(items: list[Candidate], titles: dict[str, str], scope: dict[tuple[str, str | None], str], *,
         name: str, overlap_ids: set[str], workspace: str) -> dict[str, int]:
    """``scope``: per (question, reading) the text shown on its missing-facts record."""
    api = client()
    review = ensure_dataset(api, f"{name}-review", workspace, review_settings(1))
    overlap = ensure_dataset(api, f"{name}-overlap", workspace, review_settings(2))
    missing = ensure_dataset(api, f"{name}-missing", workspace, missing_settings())
    main_items = [item for item in items if item.question_id not in overlap_ids]
    overlap_items = [item for item in items if item.question_id in overlap_ids]
    if main_items:
        review.records.log(review_records(main_items, titles))
    if overlap_items:
        overlap.records.log(review_records(overlap_items, titles))
    sections = {(item.question_id, item.reading): item for item in items}
    missing.records.log([
        {"id": f"missing-{qid}" + (f"-{rid}" if rid else ""),
         "question": question_text(item.question, item.reading_label, item.reading_gloss),
         "scope": scope.get((qid, rid), ""), "question_id": qid, "reading": rid or ""}
        for (qid, rid), item in sorted(sections.items(), key=lambda pair: (pair[0][0], pair[0][1] or ""))
    ])
    return {"review": len(main_items), "overlap": len(overlap_items), "missing": len(sections)}


DATASETS = ("review", "overlap", "missing", "screening", "judge")


def pull(*, name: str, workspace: str, suffixes: tuple[str, ...] = DATASETS) -> dict[str, list[dict[str, Any]]]:
    """Read every record's responses as plain dicts (``export`` works on these; it uses submitted ones)."""
    api = client()
    usernames = {str(user.id): user.username for user in api.users}
    out: dict[str, list[dict[str, Any]]] = {}
    for suffix in suffixes:
        dataset = api.datasets(name=f"{name}-{suffix}", workspace=workspace)
        rows: list[dict[str, Any]] = []
        if dataset is not None:
            for record in dataset.records(with_responses=True, with_suggestions=False):
                responses = [
                    {"question": response.question_name, "value": response.value,
                     "user": usernames.get(str(response.user_id), str(response.user_id)),
                     "status": str(getattr(response.status, "value", response.status))}
                    for response in record.responses
                ]
                rows.append({"id": str(record.id), "metadata": dict(record.metadata.to_dict()),
                             "responses": responses})
        out[suffix] = rows
    return out
