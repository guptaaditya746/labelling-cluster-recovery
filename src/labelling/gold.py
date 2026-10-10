"""Turn the reviews of the evidence documents (D) into gold rows the harness benchmark reads.

Decision rule for a reviewed candidate:

- ``correct`` and relevant (answers / supports)  -> the candidate triple is gold
- ``partial`` with a parseable fix, and relevant -> the fixed triple is gold
- anything else                                  -> recorded as a rejected alternative,
  with the verdict and relevance (the harness uses "wrong" ones for judged precision)

A kept triple carries ``weight`` ("vital" or "okay", default okay) and
``variants`` (other acceptable wordings) for the soft matching level.

On the overlap subset both reviewers must agree on keep/drop; disagreements
go to an adjudicator (a named user whose response decides) or stay out of the
gold and are listed in the report. The adjudicator should be a third person, or the
outcome of a recorded discussion entered under a separate user: when it is one of the
two raters, that rater's own answer wins, and the export says so loudly.

Gold triples are deduplicated by their normalised (s, p, o): the same fact kept from two
passages is one gold triple whose ``sources`` lists both passages (``source`` stays the first).

An ambiguous question's triples are reviewed per interpretation of D; its gold row has one accepted
reading per interpretation, each with its own gold triples. The scorer takes the union of the
readings' gold (AmbigQA set F1), so a run answering one of two readings recovers about half.
"""

from __future__ import annotations

import hashlib
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .candidates import Candidate
from .corpus import Corpus
from .questions import Question

KEEP_RELEVANCE = {"answers", "supports"}
WEIGHTS = {"vital", "okay"}


@dataclass
class Decision:
    candidate_id: str
    keep: bool
    triple: dict[str, str] | None
    reviewers: list[str]
    reason: str


@dataclass
class ExportReport:
    decisions: list[Decision] = field(default_factory=list)
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    unreviewed: int = 0
    adjudicated: int = 0
    adjudicated_by_a_rater: int = 0       # disagreements the adjudicator had answered as one of the raters


def snapshot() -> dict[str, Any]:
    """The evaluation snapshot the harness stamps on benchmark runs (identical defaults)."""
    return {
        "corpusSnapshotId": os.getenv("CORPUS_SNAPSHOT_ID", os.getenv("CORPUS_FINGERPRINT", "unknown")),
        "ontologyVersion": os.getenv("ONTOLOGY_VERSION", "unknown"),
        "claimSchemaVersion": "v3",
        "goldVersion": None,
        "retrievalIndexVersion": os.getenv("RETRIEVAL_INDEX_VERSION", "unknown"),
    }


def file_sha256(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return "unknown"
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def claim_parts(text: str, default_source: str | None) -> tuple[str, str, str, str, str | None] | None:
    """The structure of ``subject | relation | object [| passage_id [| vital|okay]]``: (s, p, o, source,
    weight) or None when the line cannot be used. No corpus needed, so agreement can apply the same rule."""
    parts = [part.strip() for part in text.split("|")]
    if len(parts) not in (3, 4, 5) or not all(parts[:3]):
        return None
    if len(parts) == 4 and parts[3].casefold() in WEIGHTS:   # "s | p | o | vital": weight, no passage
        parts = [*parts[:3], "", parts[3]]
    source = parts[3] if len(parts) >= 4 and parts[3] else default_source
    if not source:
        return None
    weight = parts[4].casefold() if len(parts) == 5 and parts[4] else None
    if weight is not None and weight not in WEIGHTS:
        return None
    return parts[0], parts[1], parts[2], source, weight


def parse_claim_line(corpus: Corpus, text: str, *, default_source: str | None, default_paper: str | None,
                     ) -> dict[str, str] | None:
    """Parse ``subject | relation | object [| passage_id [| vital|okay]]``.

    A subject naming the paper maps to its id. The weight is only set when given.
    """
    structure = claim_parts(text, default_source)
    if structure is None:
        return None
    subject, relation, obj, source, weight = structure
    parts = [subject, relation, obj]
    passage = corpus.passages.get(source)
    paper = str(passage["paper_id"]) if passage else default_paper
    subject = parts[0]
    if paper and corpus.resolve_paper(subject) == paper:
        subject = corpus.paper_iri(paper)
    evidence = str(passage.get("text", "")) if passage else ""
    triple = {"s": subject, "p": parts[1], "o": parts[2], "support": "human review",
              "evidence": evidence[:500], "source": source}
    if weight:
        triple["weight"] = weight
    return triple


def parse_variants(corpus: Corpus, text: object, triple: dict[str, str], paper: str | None) -> list[dict[str, str]]:
    """Other acceptable wordings: a line is either another object text or ``subject | relation | object``."""
    variants: list[dict[str, str]] = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == 3 and all(parts):
            subject = parts[0]
            if paper and corpus.resolve_paper(subject) == paper:
                subject = corpus.paper_iri(paper)
            variant = {"s": subject, "p": parts[1], "o": parts[2]}
        else:
            variant = {"s": triple["s"], "p": triple["p"], "o": line}
        if variant not in variants and (variant["s"], variant["p"], variant["o"]) != (triple["s"], triple["p"], triple["o"]):
            variants.append(variant)
    return variants


def _submitted(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_user: dict[str, dict[str, Any]] = defaultdict(dict)
    for response in record.get("responses", []):
        if response.get("status") == "submitted":
            by_user[response["user"]][response["question"]] = response.get("value")
    return dict(by_user)


def would_keep(answer: dict[str, Any], default_source: str | None = "candidate passage") -> bool:
    """The keep/drop half of the decision rule (``_decide_one``), without the corpus: agreement uses it, so
    kappa and export count a "partial" without a usable fix the same way (dropped)."""
    if answer.get("relevance") not in KEEP_RELEVANCE:
        return False
    if answer.get("verdict") == "correct":
        return True
    return answer.get("verdict") == "partial" and claim_parts(str(answer.get("correction") or ""),
                                                              default_source) is not None


def _decide_one(corpus: Corpus, candidate: Candidate, answer: dict[str, Any],
                support: str = "human review") -> tuple[bool, dict | None, str]:
    verdict, relevance = answer.get("verdict"), answer.get("relevance")
    triple = {"s": candidate.s, "p": candidate.p, "o": candidate.o, "support": support,
              "evidence": candidate.evidence[:500], "source": candidate.source}
    rejected = {**triple, "verdict": verdict, "relevance": relevance}

    def kept(item: dict[str, Any]) -> dict[str, Any]:
        # The Argilla weight wins; else a weight written in the fixed claim; else okay.
        weight = str(answer.get("weight") or item.get("weight") or "okay").casefold()
        return {**item, "weight": weight if weight in WEIGHTS else "okay",
                "variants": parse_variants(corpus, answer.get("variants"), item, candidate.paper_id)}

    if relevance not in KEEP_RELEVANCE:
        return False, rejected, f"relevance={relevance}"
    if verdict == "correct":
        return True, kept(triple), "correct"
    if verdict == "partial":
        fixed = parse_claim_line(corpus, str(answer.get("correction") or ""),
                                 default_source=candidate.source, default_paper=candidate.paper_id)
        if fixed is None:
            return False, rejected, "partial without a parseable fix"
        return True, kept({**fixed, "support": support}), "partial, fixed"
    return False, rejected, f"verdict={verdict}"


def _merge_kept(triples: list[dict[str, Any]]) -> dict[str, Any]:
    """Several reviewers kept the same triple: vital only if all say vital; variants are pooled."""
    merged = dict(triples[0])
    merged["weight"] = "vital" if all(item.get("weight") == "vital" for item in triples) else "okay"
    variants: list[dict[str, str]] = []
    for item in triples:
        variants += [variant for variant in item.get("variants", []) if variant not in variants]
    merged["variants"] = variants
    return merged


def decide(corpus: Corpus, candidates: dict[str, Candidate], reviews: dict[str, list[dict[str, Any]]],
           adjudicator: str | None = None, support: str = "human review") -> ExportReport:
    report = ExportReport()
    for suffix in ("review", "overlap"):
        for record in reviews.get(suffix, []):
            candidate = candidates.get(record["id"])
            if candidate is None:
                report.problems.append(f"{suffix}: record {record['id']} has no local candidate")
                continue
            answers = _submitted(record)
            if not answers:
                report.unreviewed += 1
                continue
            outcomes = {user: _decide_one(corpus, candidate, answer, support) for user, answer in answers.items()}
            keeps = {user: outcome[0] for user, outcome in outcomes.items()}
            kept_versions = {
                (outcome[1]["s"], outcome[1]["p"], outcome[1]["o"])
                for outcome in outcomes.values() if outcome[0] and outcome[1]
            }
            # Reviewers disagree when they differ on keep/drop, or keep different fixes.
            if len(set(keeps.values())) > 1 or len(kept_versions) > 1:
                row = {"candidate_id": candidate.candidate_id, "question_id": candidate.question_id,
                       "claim": f"{candidate.s} | {candidate.p} | {candidate.o}",
                       "responses": {user: answers[user] for user in answers}}
                if adjudicator and adjudicator in outcomes:
                    keep, triple, reason = outcomes[adjudicator]
                    # With only two answers the adjudicator is one side of the disagreement (a rater whose own
                    # answer wins); a third person's adjudication is a third answer.
                    rater = len(outcomes) <= 2
                    report.adjudicated += 1
                    report.adjudicated_by_a_rater += rater
                    report.decisions.append(Decision(
                        candidate.candidate_id, keep, triple, sorted(answers),
                        f"adjudicated by {adjudicator}" + (" (one of the two raters)" if rater else "")
                        + f": {reason}"))
                    row["resolved_by"] = adjudicator
                    row["resolved_by_a_rater"] = rater
                report.disagreements.append(row)
                continue
            keep, triple, reason = outcomes[min(outcomes)]
            if keep and len(outcomes) > 1:
                triple = _merge_kept([outcome[1] for outcome in outcomes.values() if outcome[1]])
            elif not keep and triple and len(outcomes) > 1:
                # All dropped it: record "wrong" only if every reviewer judged it wrong.
                verdicts = {outcome[1].get("verdict") for outcome in outcomes.values() if outcome[1]}
                triple = {**triple, "verdict": verdicts.pop() if len(verdicts) == 1 else "mixed"}
            report.decisions.append(Decision(candidate.candidate_id, keep, triple, sorted(answers), reason))
    return report


def missing_claims(corpus: Corpus, reviews: dict[str, list[dict[str, Any]]]) -> tuple[
        dict[tuple[str, str | None], list[dict[str, str]]], list[str]]:
    """Claims reviewers added, keyed by (question id, reading id or None): from the missing-facts records
    and from the screening check (papers marked not relevant that people found relevant)."""
    added: dict[tuple[str, str | None], list[dict[str, str]]] = defaultdict(list)
    problems: list[str] = []
    for record in [*reviews.get("missing", []), *reviews.get("screening", [])]:
        metadata = record.get("metadata", {})
        question_id = str(metadata.get("question_id") or record["id"].removeprefix("missing-"))
        reading = str(metadata.get("reading") or "") or None
        for user, answer in _submitted(record).items():
            for line in str(answer.get("missing") or "").splitlines():
                if not line.strip():
                    continue
                triple = parse_claim_line(corpus, line, default_source=None, default_paper=None)
                if triple is None or triple["source"] not in corpus.passages:
                    problems.append(f"{question_id}: unusable missing-claim line from {user}: {line!r}")
                    continue
                triple["support"] = f"added by {user}"
                triple.setdefault("weight", "okay")
                triple.setdefault("variants", [])
                added[(question_id, reading)].append(triple)
    return dict(added), problems


def build_claims(corpus: Corpus, triples: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Claims built with the harness's own constructor, so field structure matches predictions."""
    try:
        from harness_extraction.claims import build_claims as harness_build_claims
        from harness_schema.models import Triple
    except ImportError as exc:
        raise SystemExit(
            "gold_claims are built with the harness's build_claims; install the harness packages into "
            "this environment with `make harness` (HARNESS_ROOT set to the harness checkout)"
        ) from exc

    def paper_for_source(source_id: str, subject: str) -> str:
        passage = corpus.passages.get(source_id)
        return corpus.paper_iri(str(passage["paper_id"])) if passage else subject

    def evidence(triple: dict[str, Any], source: str) -> str:
        if source == triple["source"]:
            return triple["evidence"] or source
        return str((corpus.passages.get(source) or {}).get("text", ""))[:500] or source

    # A deduplicated triple stated in several passages: one claim, with every passage as a source.
    rows = [Triple(s=t["s"], p=t["p"], o=t["o"], support=t["support"], evidence=evidence(t, source), source=source)
            for t in triples for source in (t.get("sources") or [t["source"]])]
    claims, _relations = harness_build_claims(rows, paper_for_source)
    return [claim.model_dump(by_alias=True) for claim in claims]


def _norm_text(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def dedupe(triples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One gold triple per normalised (s, p, o). The same fact kept from several passages keeps every
    passage in ``sources`` (``source`` stays the first one's); it is vital if any copy is (as the scorer
    treats a fact), and the variants are pooled."""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for triple in triples:
        key = (_norm_text(triple["s"]), _norm_text(triple["p"]), _norm_text(triple["o"]))
        first = merged.get(key)
        if first is None:
            merged[key] = {**triple, "sources": list(dict.fromkeys(triple.get("sources") or [triple["source"]]))}
            continue
        for source in triple.get("sources") or [triple["source"]]:
            if source not in first["sources"]:
                first["sources"].append(source)
        if "weight" in first or "weight" in triple:
            first["weight"] = "vital" if "vital" in (first.get("weight"), triple.get("weight")) else "okay"
        if "variants" in first or "variants" in triple:
            first["variants"] = list(first.get("variants") or [])
            first["variants"] += [v for v in triple.get("variants") or [] if v not in first["variants"]]
    return list(merged.values())


def gold_row(question: Question, triples: list[dict[str, str]], *, corpus: Corpus, version: str,
             ontology_hash: str, reviewers: list[str], expected_status: str = "answered",
             rejected: list[dict[str, str]] | None = None, notes: str = "",
             accepted_readings: list[dict[str, Any]] | None = None,
             label_source: str = "people") -> dict[str, Any]:
    """``accepted_readings``: one ``{label, measure, reading_id, gold_triples}`` per reviewed interpretation
    of an ambiguous question (its own ``triples`` are then empty)."""
    triples = dedupe(triples)
    every = triples + [item for reading in accepted_readings or [] for item in reading["gold_triples"]]
    now = datetime.now(UTC).isoformat()
    complexity = question.complexity.as_dict() if question.complexity else None
    return {
        "id": f"gold-{version}-{question.id}",
        "question_id": question.id,
        "question": question.question,
        "category": question.category,
        "complexity": complexity,
        "difficulty": complexity["level"] if complexity else None,
        "origin": question.origin,
        "status": "approved",
        "ambiguous": question.category == "ambiguous",
        "accepted_readings": accepted_readings or [],
        "gold_triples": triples,
        "gold_claims": build_claims(corpus, triples) if triples else [],
        "evidence_refs": sorted({source for triple in every for source in triple.get("sources") or [triple["source"]]}),
        "paper_ids": sorted({triple["s"] for triple in every if triple["s"].startswith(("p", "https://dblp"))}),
        "expected_status": expected_status,
        "gold_version": version,
        "corpus_fingerprint": corpus.fingerprint(),
        "ontology_hash": ontology_hash,
        "snapshot": snapshot(),
        "reviewer": ", ".join(reviewers),
        "reviewed_at": now,
        "created_by": "labelling",
        "annotators": [{"source": "labelling", "reviewer": name} for name in reviewers],
        "readings": [],
        "review_history": [{"at": now, "actor": "labelling", "status": "approved", "comment": notes}],
        "llm_proposals": [],
        "forbidden_alternatives": rejected or [],
        "run_id": None,
        "notes": notes,
        "label_source": label_source,
    }


def review_rows(questions: list[Question], candidates: list[Candidate], report: ExportReport,
                added: dict[tuple[str, str | None], list[dict[str, str]]], *, corpus: Corpus, version: str,
                ontology_hash: str, notes: str = "evidence document reviewed in Argilla",
                label_source: str = "people") -> list[dict[str, Any]]:
    """One gold row per question with kept triples: the answer key of its evidence document."""
    by_candidate = {item.candidate_id: item for item in candidates}
    kept: dict[tuple[str, str | None], list[dict[str, str]]] = defaultdict(list)
    rejected: dict[str, list[dict[str, str]]] = defaultdict(list)
    reviewers: dict[str, set[str]] = defaultdict(set)
    readings: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)     # question -> id -> (label, gloss)
    for item in candidates:
        if item.reading:
            readings[item.question_id][item.reading] = (item.reading_label or item.reading, item.reading_gloss or "")
    for decision in report.decisions:
        candidate = by_candidate[decision.candidate_id]
        reviewers[candidate.question_id].update(decision.reviewers)
        if decision.keep and decision.triple:
            kept[(candidate.question_id, candidate.reading)].append(decision.triple)
        elif decision.triple:
            rejected[candidate.question_id].append(decision.triple)
    rows = []
    for question in questions:
        accepted = [
            {"label": label, "measure": gloss, "reading_id": rid,
             "gold_triples": dedupe(kept.get((question.id, rid), []) + added.get((question.id, rid), []))}
            for rid, (label, gloss) in sorted(readings.get(question.id, {}).items())
        ]
        accepted = [reading for reading in accepted if reading["gold_triples"]]
        triples = kept.get((question.id, None), []) + added.get((question.id, None), [])
        if not triples and not accepted:
            continue
        rows.append(gold_row(question, triples, corpus=corpus, version=version, ontology_hash=ontology_hash,
                             reviewers=sorted(reviewers.get(question.id, set())) or ["unknown"],
                             rejected=rejected.get(question.id, []), notes=notes,
                             accepted_readings=accepted, label_source=label_source))
    return rows


# ─── Agreement ──────────────────────────────────────────────────────────────

KAPPA_BAR = 0.61                 # "substantial" (Landis and Koch 1977): the bar for letting soft use the judge
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1998


def kappa_table(table: Mapping[tuple[Any, Any], float]) -> float | None:
    """Cohen's kappa from a (possibly weighted) 2-rater table; None when undefined (empty, no variation)."""
    total = sum(table.values())
    if total <= 0:
        return None
    observed = sum(weight for (left, right), weight in table.items() if left == right) / total
    left_sums: dict[Any, float] = defaultdict(float)
    right_sums: dict[Any, float] = defaultdict(float)
    for (left, right), weight in table.items():
        left_sums[left] += weight
        right_sums[right] += weight
    expected = sum(weight * right_sums.get(label, 0.0) for label, weight in left_sums.items()) / (total * total)
    if expected >= 1 - 1e-12:
        return None
    return (observed - expected) / (1 - expected)


def cohen_kappa(pairs: list[tuple[Any, Any]], weights: list[float] | None = None) -> float | None:
    """Cohen's kappa for two raters; ``weights`` (one per pair) estimate a population table from a
    stratified sample. None when undefined (no items or no variation)."""
    if not pairs:
        return None
    table: dict[tuple[Any, Any], float] = defaultdict(float)
    for index, pair in enumerate(pairs):
        table[pair] += 1.0 if weights is None else weights[index]
    return kappa_table(table)


def _quantile(values: list[float], q: float) -> float:
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap_kappa(groups: Mapping[str, Counter], resamples: int = BOOTSTRAP_RESAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """95% percentile interval for kappa, resampling whole questions (``groups``: question -> pair counts),
    since records of one question are not independent. Resamples where kappa is undefined are skipped and
    counted. Fewer than two questions: no interval."""
    keys = sorted(groups)
    if len(keys) < 2:
        return {"interval": None, "questions": len(keys), "resamples": 0, "undefined": 0}
    rng = random.Random(seed)
    values: list[float] = []
    undefined = 0
    for _ in range(resamples):
        table: Counter = Counter()
        for _ in keys:
            table.update(groups[rng.choice(keys)])
        value = kappa_table(table)
        if value is None:
            undefined += 1
        else:
            values.append(value)
    values.sort()
    interval = [round(_quantile(values, 0.025), 4), round(_quantile(values, 0.975), 4)] if values else None
    return {"interval": interval, "questions": len(keys), "resamples": resamples, "undefined": undefined}


def agreement(reviews: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Kappa on the overlap dataset for the two most active reviewers (ties broken by name).

    Keep/drop uses the export's own rule (``would_keep``): a "partial" without a usable fix is a drop.
    Records answered by one person only, or by a pair other than the two most active, are left out and
    counted. ``keep_kappa_95`` and ``verdict_kappa_95`` resample questions (``bootstrap_kappa``).
    """
    records = [(record, _submitted(record)) for record in reviews.get("overlap", [])]
    answered = [(record, answers) for record, answers in records if answers]
    activity = Counter(user for _, answers in answered for user in answers)
    if len(activity) < 2:
        return {"reviewers": sorted(activity), "items": 0, "records_with_responses": len(answered),
                "verdict_kappa": None, "keep_kappa": None}
    first, second = sorted(activity, key=lambda user: (-activity[user], user))[:2]
    both = [(str(record.get("metadata", {}).get("question_id") or record["id"]), answers[first], answers[second])
            for record, answers in answered if first in answers and second in answers]
    single = sum(1 for _, answers in answered if len(answers) == 1)

    def keep(answer: dict[str, Any]) -> str:
        return "keep" if would_keep(answer) else "drop"

    keep_groups: dict[str, Counter] = defaultdict(Counter)
    verdict_groups: dict[str, Counter] = defaultdict(Counter)
    for question_id, a, b in both:
        keep_groups[question_id][(keep(a), keep(b))] += 1
        verdict_groups[question_id][(a.get("verdict"), b.get("verdict"))] += 1
    keep_interval = bootstrap_kappa(keep_groups)
    both_kept = [(a, b) for _, a, b in both if keep(a) == keep(b) == "keep"]
    return {
        "reviewers": [first, second],
        "items": len(both),
        "questions": len(keep_groups),
        "records_with_responses": len(answered),
        "records_dropped_one_reviewer": single,
        "records_dropped_other_reviewers": len(answered) - len(both) - single,
        "weight_kappa": cohen_kappa([(a.get("weight") or "okay", b.get("weight") or "okay") for a, b in both_kept]),
        "verdict_kappa": cohen_kappa([(a.get("verdict"), b.get("verdict")) for _, a, b in both]),
        "relevance_kappa": cohen_kappa([(a.get("relevance"), b.get("relevance")) for _, a, b in both]),
        "keep_kappa": cohen_kappa([(keep(a), keep(b)) for _, a, b in both]),
        "raw_keep_agreement": (sum(keep(a) == keep(b) for _, a, b in both) / len(both)) if both else None,
        "keep_kappa_95": keep_interval["interval"],
        "verdict_kappa_95": bootstrap_kappa(verdict_groups)["interval"],
        "bootstrap": {**{k: v for k, v in keep_interval.items() if k != "interval"},
                      "method": "percentile, resampling questions", "seed": BOOTSTRAP_SEED},
    }


def _complete(record: dict[str, Any]) -> bool:
    return len(_submitted(record)) >= 2


def overlap_from_snapshots(snapshots: list[dict[str, Any]], live: list[dict[str, Any]],
                           ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The independent round of the overlap: per record, its version in the earliest snapshot where two
    people had submitted it; records never seen complete in a snapshot come from ``live``.

    ``snapshots`` are ``{"taken_at", "path", "overlap": [records]}``, oldest first. Adjudication that edits
    a rater's response after that snapshot no longer changes kappa.
    """
    chosen: dict[str, dict[str, Any]] = {}
    for snap in snapshots:
        for record in snap.get("overlap") or []:
            if record["id"] not in chosen and _complete(record):
                chosen[record["id"]] = record
    live_ids = {record["id"] for record in live}
    merged = [chosen.get(record["id"], record) for record in live]
    merged += [record for rid, record in chosen.items() if rid not in live_ids]
    return merged, {
        "snapshots": [snap.get("path") or snap.get("taken_at") for snap in snapshots],
        "records_from_snapshots": len(chosen),
        "records_from_live": sum(1 for record in live if record["id"] not in chosen and _complete(record)),
    }


def judge_agreement(reviews: dict[str, list[dict[str, Any]]], manifest: dict[str, Any] | None = None,
                    ) -> dict[str, Any]:
    """People's "same fact?" answers against the harness judge's hidden answer (the ``-judge`` dataset).

    The sample is stratified by the judge's answer (``labelling judge-push`` draws about half "yes" and
    half "no"), and kappa depends on prevalence, so a raw-sample kappa is not the population's. With the
    push's manifest (population sizes N_yes, N_no) each record is weighted by N_h / n_h (n_h: answered
    records of its stratum), and kappa is computed from that weighted table. The decision uses the
    weighted kappa: 0.61 or more ("substantial", Landis and Koch 1977) lets the soft level use the judge;
    below it the soft level relies on reviewer variants only. Without a manifest (an older push) the raw
    kappa decides, with a warning.

    One response per record for the overall numbers: the alphabetically first user who answered it (a
    fixed rule that does not look at the answer). ``per_person`` gives each person's own kappa against the
    judge. ``precision_judge_yes`` = P(person yes | judge yes) and ``person_yes_given_judge_no`` = P(person
    yes | judge no), each with a 95% Wilson interval: the first is how often a judged soft credit is right,
    the second how many soft credits the judge withholds.
    """
    from .triage import wilson

    warnings: list[str] = []
    allowed: set[str] | None = None
    population: dict[str, int] | None = None
    if manifest:
        allowed = {str(rid) for ids in (manifest.get("record_ids") or {}).values() for rid in ids} or None
        population = {label: int((manifest.get("population") or {}).get(label, 0)) for label in ("yes", "no")}
    else:
        warnings.append("no judge-push manifest (a push made before it was recorded): kappa is computed on the "
                        "raw sample, whose yes/no mix is set by the sampler, not by the population; run "
                        "`labelling judge-push` again to record the population sizes")
    rows: list[tuple[str, str, dict[str, str]]] = []
    ignored = 0
    for record in reviews.get("judge", []):
        judged = str(record.get("metadata", {}).get("judge") or "")
        if judged not in {"yes", "no"}:
            continue
        record_id = str(record.get("id", ""))
        if allowed is not None and record_id not in allowed:
            ignored += 1
            continue
        answers = {user: str(answer["same_fact"]) for user, answer in _submitted(record).items()
                   if answer.get("same_fact") in {"yes", "no"}}
        if answers:
            rows.append((record_id, judged, answers))
    if ignored:
        warnings.append(f"{ignored} judge records are not in the latest push's manifest (an earlier push or "
                        "prompt) and were left out")

    def weights_for(pairs: list[tuple[str, str]]) -> list[float] | None:
        if population is None:
            return None
        sampled = Counter(judged for _, judged in pairs)
        return [population[judged] / sampled[judged] for _, judged in pairs]

    def summary(pairs: list[tuple[str, str]]) -> dict[str, Any]:
        weights = weights_for(pairs)
        return {"items": len(pairs), "raw_kappa": cohen_kappa(pairs),
                "weighted_kappa": cohen_kappa(pairs, weights) if weights is not None else None}

    one = [(answers[min(answers)], judged) for _, judged, answers in rows]
    weights = weights_for(one)
    raw = cohen_kappa(one)
    weighted = cohen_kappa(one, weights) if weights is not None else None
    if population is not None:
        for label in ("yes", "no"):
            if population[label] and not any(judged == label for _, judged in one):
                warnings.append(f"no answered record where the judge said {label!r}: the weighted table "
                                "misses that stratum")
    kappa = weighted if population is not None else raw

    def conditional(label: str) -> dict[str, Any]:
        stratum = [person for person, judged in one if judged == label]
        hits = sum(person == "yes" for person in stratum)
        low, high = wilson(hits, len(stratum))
        return {"rate": round(hits / len(stratum), 4) if stratum else None, "n": len(stratum),
                "ci95": [round(low, 4), round(high, 4)]}

    users = sorted({user for _, _, answers in rows for user in answers})
    total = sum(weights) if weights else 0.0
    return {
        "items": len(one),
        "responses": sum(len(answers) for _, _, answers in rows),
        "one_response_rule": "per record, the alphabetically first user who answered",
        "kappa": kappa,
        "kappa_basis": "population-weighted (N_h/n_h)" if population is not None else "raw sample (no manifest)",
        "raw_kappa": raw,
        "weighted_kappa": weighted,
        "raw_agreement": (sum(a == b for a, b in one) / len(one)) if one else None,
        "weighted_agreement": (sum(w for (a, b), w in zip(one, weights, strict=True) if a == b) / total
                               if weights and total else None),
        "population": population,
        "sample": dict(Counter(judged for _, judged in one)),
        "precision_judge_yes": conditional("yes"),
        "person_yes_given_judge_no": conditional("no"),
        "judge_says_yes_people_say_no": sum(a == "no" and b == "yes" for a, b in one),
        "judge_says_no_people_say_yes": sum(a == "yes" and b == "no" for a, b in one),
        "per_person": {user: summary([(answers[user], judged) for _, judged, answers in rows if user in answers])
                       for user in users},
        "bar": KAPPA_BAR,
        "use_judge": kappa is not None and kappa >= KAPPA_BAR,
        "warnings": warnings,
    }
