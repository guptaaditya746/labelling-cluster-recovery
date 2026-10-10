"""Triage: a model sorts every triple of an evidence document (D) by how it relates to the question.

Reviewing every triple of D does not scale (about 1,000 triples per 20 relevant papers), so review is
stratified: people label every triple the model put in ``answers`` or ``context``, plus a random sample of
the ``irrelevant`` ones (``labelling pool --sample``). The sample measures what triage missed;
``labelling export`` reports that miss rate and the recall of the reviewed tiers, each with a 95% interval. The labels only order the review: they are never gold, and
reviewers do not see them. The exception is ``labelling draft``: a stand-in answer key from the labels
alone, marked as a draft, for trying the harness before the review is done (``draft_rows``).

Labels go to ``<docs>/<question id>/triage.jsonl`` (one line per interpretation and triple), so a stopped
run resumes and a finished one is not repeated. The model is ``TRIAGE_MODEL`` (default the harness's
high model).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

LABELS = ("answers", "context", "irrelevant")
PROMPT = Path(__file__).with_name("prompts") / "triage.md"


def triage_file(docs: Path, question_id: str) -> Path:
    return Path(docs).expanduser() / question_id / "triage.jsonl"


def load_labels(docs: Path, question_id: str) -> dict[tuple[str, str], str]:
    """(interpretation id, triple id) -> label, from the triage file (the last line for a key wins)."""
    path = triage_file(docs, question_id)
    labels: dict[tuple[str, str], str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                labels[(row["reading"], row["triple"])] = row["label"]
    return labels


def default_model() -> Any:
    """The harness model client for TRIAGE_MODEL (default HARNESS_MODEL_HIGH)."""
    from harness_config import settings
    from harness_extraction import OpenAICompatibleModel

    from .doc_settings import doc_settings

    alias = os.environ.get("TRIAGE_MODEL", "").strip() or settings().model_high
    return OpenAICompatibleModel(model=alias, timeout_s=doc_settings().call_timeout_s)


def _evidence(triple: dict[str, Any]) -> str:
    return " … ".join(str(source.get("evidence") or "")[:300] for source in (triple.get("sources") or [])[:2])


def _ask(model: Any, question: str, reading: dict[str, Any] | None, batch: list[dict[str, Any]]) -> dict[str, str]:
    payload = {
        "question": question,
        "interpretation": {"label": reading.get("label"), "gloss": reading.get("gloss")} if reading else {},
        "facts": [{"id": t["id"], "subject": t["s"], "relation": t["p"], "object": t["o"],
                   "paper": t.get("paper_id"), "passage": _evidence(t)} for t in batch],
    }
    reply = model.complete_json(PROMPT.read_text(encoding="utf-8"), json.dumps(payload, ensure_ascii=False))
    wanted = {t["id"] for t in batch}
    out: dict[str, str] = {}
    for item in (reply or {}).get("labels") or []:
        if isinstance(item, dict) and item.get("id") in wanted and item.get("label") in LABELS:
            out[str(item["id"])] = str(item["label"])
    return out


def triage_doc(docs: Path, question_id: str, model: Any, *, batch: int = 20, workers: int = 4,
               log: Callable[[str], None] = print) -> Counter:
    """Label every triple of one document that has no label yet; returns the label counts."""
    doc = json.loads((Path(docs).expanduser() / question_id / "doc.json").read_text(encoding="utf-8"))
    question = str(doc["question"]["question"])
    sections = doc.get("interpretations") or []
    ambiguous = len(sections) > 1 or doc["question"].get("category") == "ambiguous"
    known = load_labels(docs, question_id)
    jobs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for reading in sections:
        todo = [t for t in reading.get("triples") or [] if (reading["id"], t["id"]) not in known]
        jobs += [(reading, todo[i:i + batch]) for i in range(0, len(todo), batch)]
    total = sum(len(items) for _, items in jobs)
    log(f"{question_id}: {total} triples to sort ({len(known)} already sorted), {len(jobs)} calls")
    path = triage_file(docs, question_id)
    done = 0

    def run(job: tuple[dict[str, Any], list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, str]]:
        reading, items = job
        try:
            return reading, _ask(model, question, reading if ambiguous else None, items)
        except Exception as exc:  # noqa: BLE001 - a failed batch stays unsorted and is retried next time
            log(f"{question_id}: a batch failed ({type(exc).__name__}: {str(exc)[:120]}); it is retried next run")
            return reading, {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, path.open("a", encoding="utf-8") as handle:
        for reading, labels in pool.map(run, jobs):
            for triple_id, label in labels.items():
                handle.write(json.dumps({"reading": reading["id"], "triple": triple_id, "label": label}) + "\n")
                known[(reading["id"], triple_id)] = label
            handle.flush()
            done += len(labels)
            if jobs and done and done % max(batch * 10, 1) < len(labels):
                log(f"{question_id}: {done}/{total} sorted")
    counts = Counter(known.values())
    unsorted = sum(len(r.get("triples") or []) for r in sections) - len(known)
    log(f"{question_id}: " + ", ".join(f"{label} {counts.get(label, 0)}" for label in LABELS)
        + (f", unsorted {unsorted} (run again)" if unsorted else ""))
    return counts


def sample_ids(ids: Iterable[str], size: int, seed: str) -> set[str]:
    """A deterministic random sample: the same seed and ids give the same sample."""
    ranked = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    return set(ranked[:max(0, size)])


def stratify(items: list[Any], labels: dict[str, dict[tuple[str, str], str]], *, tiers: Iterable[str],
             sample: int, seed: str) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """The review candidates: per question, every triple triage put in ``tiers`` (or left unsorted) and a
    random ``sample`` of the rest. ``labels`` maps a question id to its triage labels; a question without
    them keeps all its triples. Within a question the order is shuffled, so the sampled records are not
    grouped where a reviewer could notice them. Returns the candidates and the strata sizes per question
    (``labelling export`` needs them to estimate what triage missed)."""
    tiers = set(tiers)
    groups: dict[str, list[Any]] = {}
    for item in items:
        groups.setdefault(item.question_id, []).append(item)
    chosen: list[Any] = []
    strata: dict[str, dict[str, Any]] = {}
    for question_id, group in groups.items():
        known = labels.get(question_id) or {}
        if not known:
            chosen += group
            strata[question_id] = {"triaged": False, "reviewed": len(group)}
            continue
        by_triple = {triple: label for (_, triple), label in known.items()}   # one interpretation: no reading
        reviewed, rest = [], []
        for item in group:
            tag = known.get((item.reading, item.triple_id)) if item.reading else by_triple.get(item.triple_id)
            (rest if tag is not None and tag not in tiers else reviewed).append(item.model_copy(update={"triage": tag}))
        picked = sample_ids([item.candidate_id for item in rest], sample, f"{seed}:{question_id}")
        keep = ([item.model_copy(update={"stratum": "triaged"}) for item in reviewed]
                + [item.model_copy(update={"stratum": "sample"}) for item in rest if item.candidate_id in picked])
        chosen += sorted(keep, key=lambda item: hashlib.sha256(f"{seed}:{item.candidate_id}".encode()).hexdigest())
        strata[question_id] = {
            "triaged": True, "tiers": sorted(tiers), "labels": dict(Counter(known.values())),
            "reviewed_tiers": len(reviewed), "unsorted": sum(item.triage is None for item in reviewed),
            "rest": len(rest), "sampled": len(picked), "seed": seed,
        }
    return chosen, strata


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (sound for small samples and proportions near 0)."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return max(0.0, centre - half), min(1.0, centre + half)


def _recall(tier_kept: int, sample_kept: int, missed: float) -> float | None:
    denominator = tier_kept + sample_kept + missed
    return round(tier_kept / denominator, 4) if denominator else None


def miss_report(items: list[Any], decisions: list[Any], strata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per triaged question: how many gold triples triage put outside the reviewed tiers.

    The kept share of the reviewed sample, times the size of the rest, estimates the gold triples that
    were never reviewed (with a 95% Wilson interval). ``estimated_recall`` is the share of all gold
    triples the reviewed tiers hold; the sample's own kept triples are in the gold either way.
    ``estimated_recall_95`` puts the Wilson bounds into the same ratio:
    tiers_kept / (tiers_kept + sample_kept + N x [high, low]), N the unreviewed rest. With no kept triple
    in the sample the point estimate is 1.0 but the interval is not: its lower end is what the sample
    cannot rule out.

    Only decided records count. Tier records still unreviewed or in an unresolved disagreement
    (``tiers_unresolved``) are missing from ``tiers_kept``, which biases the recall estimate down;
    unresolved sample records (``sample_unresolved``) shrink the sample and widen the interval, and they
    stay in N.
    """
    by_id = {item.candidate_id: item for item in items}
    out: dict[str, Any] = {}
    for question_id, sizes in sorted(strata.items()):
        if not sizes.get("triaged"):
            continue
        counts = Counter()
        for decision in decisions:
            item = by_id.get(decision.candidate_id)
            if item is None or item.question_id != question_id:
                continue
            counts[f"{item.stratum}_reviewed"] += 1
            counts[f"{item.stratum}_kept"] += bool(decision.keep)
        trials, kept = counts["sample_reviewed"], counts["sample_kept"]
        unreviewed_rest = sizes["rest"] - trials
        low, high = wilson(kept, trials)
        estimate = unreviewed_rest * kept / trials if trials else None
        tier_kept = counts["triaged_kept"]

        bounds = [_recall(tier_kept, kept, unreviewed_rest * high), _recall(tier_kept, kept, unreviewed_rest * low)]
        out[question_id] = {
            "reviewed_tiers": sizes["reviewed_tiers"], "tiers_reviewed": counts["triaged_reviewed"],
            "tiers_unresolved": sizes["reviewed_tiers"] - counts["triaged_reviewed"],
            "tiers_kept": tier_kept, "rest": sizes["rest"], "sampled": sizes["sampled"],
            "sample_reviewed": trials, "sample_unresolved": sizes["sampled"] - trials, "sample_kept": kept,
            "sample_kept_share": round(kept / trials, 4) if trials else None,
            "estimated_missed": round(estimate, 1) if estimate is not None else None,
            "estimated_missed_95": [round(unreviewed_rest * low, 1), round(unreviewed_rest * high, 1)],
            "estimated_recall": _recall(tier_kept, kept, estimate) if estimate is not None else None,
            "estimated_recall_95": bounds if None not in bounds else None,
        }
    return out


DRAFT_REVIEWER = "triage-draft"
DRAFT_RELEVANCE = {"answers": "answers", "context": "supports"}


def draft_rows(corpus: Any, root: Path, questions: list[Any], *, tiers: Iterable[str], version: str,
               ontology_hash: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """A stand-in answer key until people have reviewed D: every triple triage put in ``tiers`` is kept as
    if a reviewer had marked it correct (answers -> answers, context -> supports), weight okay.

    Nothing here checks that a triple is correct. The rows have the reviewed rows' ids
    (``gold-<version>-<question>``), so installing the reviewed export replaces them, and are marked as
    drafts: ``gold_version`` ``<version>-draft``, reviewer ``triage-draft``. Returns the rows and, per
    question left out, the reason.
    """
    from . import candidates as cand
    from . import gold

    items, missing = cand.from_evidence_docs(corpus, root, questions)
    skipped = {qid: "no evidence document" for qid in missing}
    tiers = set(tiers)
    labels = {qid: load_labels(root, qid) for qid in {item.question_id for item in items}}
    report = gold.ExportReport()
    for question_id, known in labels.items():
        if not known:
            skipped[question_id] = "not triaged (labelling triage)"
    by_triple = {qid: {triple: label for (_, triple), label in known.items()} for qid, known in labels.items()}
    for item in items:
        known = labels[item.question_id]
        tag = known.get((item.reading, item.triple_id)) if item.reading else by_triple[item.question_id].get(
            item.triple_id)
        if tag not in tiers:
            continue
        keep, triple, _ = gold._decide_one(
            corpus, item, {"verdict": "correct", "relevance": DRAFT_RELEVANCE.get(tag, "supports")},
            support="triage draft, not reviewed")
        report.decisions.append(gold.Decision(item.candidate_id, keep, triple, [DRAFT_REVIEWER], f"triage {tag}"))
    rows = gold.review_rows(questions, items, report, {}, corpus=corpus, version=version,
                            ontology_hash=ontology_hash, label_source="model-triage",
                            notes="DRAFT: triage labels, not reviewed by people; replace with the reviewed export")
    for row in rows:
        row["gold_version"] = f"{version}-draft"
    return rows, skipped


def is_draft(row: dict[str, Any]) -> bool:
    return str(row.get("gold_version") or "").endswith("-draft") or row.get("reviewer") == DRAFT_REVIEWER
