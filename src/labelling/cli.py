"""labelling: build the gold set: evidence documents (D), reviewed once (see README for the workflow).

  validate       check the question set and print the corpus fingerprint
  evidence-docs  build one evidence document (D) per question: every paper screened (docs/EVIDENCE_DOCS.md)
  triage         a model sorts each D triple: answers, context or irrelevant (orders review, never gold)
  pool           the triples of the evidence documents as review candidates (triaged tiers + a sample)
  draft          a stand-in answer key from triage alone (marked draft; the reviewed export replaces it)
  push           load them into Argilla (review, overlap, missing datasets)
  screening-sample  a random sample of the papers D marked not relevant, per question (to check screening)
  screening-push    load that sample into Argilla (the screening dataset)
  screening-report  the screening's estimated false-negative rate, from people's answers
  snapshot       save the overlap responses now (do it after the independent round, before adjudicating)
  export         reviews -> gold rows in the harness format (the answer keys), with agreement report
  install        merge gold rows into the harness datasets/gold/candidates.jsonl
  judge-push     sample the harness judge's cached "same fact?" decisions into Argilla for people
  judge-agreement  kappa between people and the judge (decides whether soft scores may use it)

Metrics, their reasons and limits: docs/EVALUATION.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import candidates as cand
from . import gold
from .corpus import Corpus, read_jsonl
from .doc_settings import doc_settings
from .questions import LEVELS, load_questions


def load_env(path: Path = Path(".env")) -> None:
    """Minimal .env reader; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            if key and value and key not in os.environ:
                os.environ[key] = os.path.expanduser(value)


def env_path(name: str, default: str) -> Path:
    return Path(os.path.expanduser(os.getenv(name, default)))


def ontology_files() -> list[Path]:
    """The harness seed: the DBLP schema and, since the CS-KG content core, its subset file."""
    seed = env_path("HARNESS_ROOT", "../rambodambo") / "seed"
    return [seed / "dblp_schema_2024-06-14.ttl", seed / "cskg_ontology_2.0_subset.ttl"]


def ontology_hash() -> str:
    """SHA-256 over the seed files' own hashes; "unknown" when one is missing (check HARNESS_ROOT)."""
    parts = [gold.file_sha256(path) for path in ontology_files()]
    if "unknown" in parts:
        return "unknown"
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def overlap_ids(questions, spec: str, version: str) -> set[str]:
    synthesis = [item.id for item in questions]
    if spec == "none":
        return set()
    if spec != "auto":
        chosen = {part.strip() for part in spec.split(",") if part.strip()}
        unknown = chosen - set(synthesis)
        if unknown:
            raise SystemExit(f"overlap ids are not in the question set: {sorted(unknown)}")
        return chosen
    # Deterministic 20% sample, stable for a given gold version.
    ranked = sorted(synthesis, key=lambda qid: hashlib.sha256(f"{version}:{qid}".encode()).hexdigest())
    return set(ranked[: max(1, math.ceil(0.2 * len(synthesis)))]) if synthesis else set()


def cmd_validate(args) -> int:
    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    questions = load_questions(args.questions)
    problems = []
    counts = Counter(question.category for question in questions)
    print(f"{len(questions)} questions: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for category in ("corpus_wide", "ambiguous"):
        if counts.get(category, 0) < 10:
            problems.append(f"category {category} has {counts.get(category, 0)} questions; the design asks for 10")
    levels = Counter(question.level or "unset" for question in questions)
    print("complexity: " + ", ".join(f"{level}={levels.get(level, 0)}" for level in (*LEVELS, "unset")))
    if levels.get("unset"):
        # The harness stratifies the benchmark by complexity level (GoldCandidateStore.validate_benchmark).
        print(f"note  {levels['unset']} questions have no complexity yet: rate them (two raters) before export")
    origins = Counter(question.origin or "unset" for question in questions)
    print("origin: " + ", ".join(f"{name}={origins.get(name, 0)}" for name in ("professor", "own", "unset")))
    for problem in problems:
        print(f"FAIL  {problem}")
    print(f"corpus fingerprint: {corpus.fingerprint()}")
    return 1 if problems else 0


def docs_root(value: str | None) -> Path:
    return Path(value).expanduser() if value else doc_settings().root or Path("evidence_docs")


def cmd_triage(args) -> int:
    from . import triage

    root = docs_root(args.docs)
    only = {part.strip() for part in (args.only or "").split(",") if part.strip()}
    found = sorted(path.parent.name for path in root.glob("*/doc.json"))
    chosen = [qid for qid in found if not only or qid in only]
    if only - set(found):
        print(f"no evidence document for: {sorted(only - set(found))} (in {root})")
    if not chosen:
        print(f"nothing to triage in {root}")
        return 1
    model = triage.default_model()
    for question_id in chosen:
        triage.triage_doc(root, question_id, model, batch=args.batch, workers=args.workers)
    return 0


def cmd_pool(args) -> int:
    from . import triage

    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    questions = load_questions(args.questions)
    root = docs_root(args.docs)
    items, missing = cand.from_evidence_docs(corpus, root, questions)
    total = Counter(item.question_id for item in items)
    labels = {} if args.all else {qid: triage.load_labels(root, qid) for qid in total}
    tiers = [part.strip() for part in args.tiers.split(",") if part.strip()]
    if unknown := set(tiers) - set(triage.LABELS):
        raise SystemExit(f"unknown tiers {sorted(unknown)}; choose from {triage.LABELS}")
    items, strata = triage.stratify(items, labels, tiers=tiers, sample=args.sample, seed=args.seed)
    cand.write_jsonl(args.out, items)
    Path(args.out).with_suffix(".strata.json").write_text(json.dumps(strata, indent=2))
    per_question = Counter(item.question_id for item in items)
    for question in questions:
        if question.id in missing:
            print(f"{question.id}: no evidence document: labelling evidence-docs")
            continue
        sizes = strata.get(question.id) or {}
        if sizes.get("triaged"):
            print(f"{question.id}: {per_question[question.id]} of {total[question.id]} triples to review: "
                  f"{sizes['reviewed_tiers']} in {'+'.join(tiers)}"
                  + (f" ({sizes['unsorted']} unsorted)" if sizes["unsorted"] else "")
                  + f", {sizes['sampled']} sampled from {sizes['rest']} others")
        elif question.id in total:
            print(f"{question.id}: {per_question[question.id]} triples"
                  + ("" if args.all else "  <- not triaged, all triples go to review (labelling triage)"))
    print(f"wrote {len(items)} candidates to {args.out}")
    return 0


def cmd_draft(args) -> int:
    from . import triage

    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    questions = load_questions(args.questions)
    tiers = [part.strip() for part in args.tiers.split(",") if part.strip()]
    rows, skipped = triage.draft_rows(corpus, docs_root(args.docs), questions, tiers=tiers, version=args.version,
                                      ontology_hash=ontology_hash())
    cand.write_jsonl(args.out, rows)
    for row in rows:
        count = len(row["gold_triples"]) + sum(len(r["gold_triples"]) for r in row["accepted_readings"])
        print(f"{row['question_id']}: {count} draft gold triples")
    for question_id, reason in sorted(skipped.items()):
        print(f"{question_id}: skipped, {reason}")
    print(f"wrote {len(rows)} DRAFT gold rows to {args.out} (gold_version {args.version}-draft, not reviewed);"
          f" install with: labelling install --gold {args.out}")
    return 0


def review_scope(root: Path, items: list[cand.Candidate]) -> dict[tuple[str, str | None], str]:
    """What the missing-facts record shows: the papers D found relevant, per question and reading."""
    scope: dict[tuple[str, str | None], str] = {}
    for question_id in sorted({item.question_id for item in items}):
        path = root / question_id / "doc.json"
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        sections = doc.get("interpretations") or []
        for reading in sections:
            relevant = [row["paper_id"] for row in reading.get("coverage") or [] if row.get("status") == "relevant"]
            key = (question_id, str(reading["id"]) if len(sections) > 1 or doc.get("question", {}).get(
                "category") == "ambiguous" else None)
            scope[key] = (f"{len(relevant)} papers found relevant: " + ", ".join(f"`{paper}`" for paper in relevant)
                          if relevant else "No paper was found relevant.")
    return scope


def cmd_push(args) -> int:
    from . import argilla_io

    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    questions = load_questions(args.questions)
    items = cand.read_candidates(args.candidates)
    titles = {pid: corpus.title(pid) for pid in corpus.papers}
    chosen = overlap_ids(questions, args.overlap, args.name)
    counts = argilla_io.push(items, titles, review_scope(docs_root(args.docs), items), name=args.name,
                             overlap_ids=chosen, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"))
    print(f"overlap questions: {sorted(chosen)}")
    print(json.dumps(counts))
    return 0


# ─── Overlap snapshots (the independent round, before adjudication) ─────────

def save_overlap_snapshot(folder: Path, name: str, reviews: dict, taken_by: str) -> Path:
    """A timestamped copy of the overlap responses; agreement is computed from the earliest ones."""
    now = datetime.now(UTC)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-overlap-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    path.write_text(json.dumps({"name": name, "taken_at": now.isoformat(), "taken_by": taken_by,
                                "overlap": reviews.get("overlap", [])}, indent=2, default=str))
    return path


def load_overlap_snapshots(folder: Path, name: str) -> list[dict]:
    snapshots = []
    for path in sorted(Path(folder).glob(f"{name}-overlap-*.json")):
        data = json.loads(path.read_text())
        snapshots.append({**data, "path": str(path)})
    return sorted(snapshots, key=lambda snap: str(snap.get("taken_at")))


def cmd_snapshot(args) -> int:
    from . import argilla_io

    reviews = argilla_io.pull(name=args.name, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"),
                              suffixes=("overlap",))
    path = save_overlap_snapshot(Path(args.dir), args.name, reviews, taken_by="snapshot")
    records = reviews.get("overlap", [])
    complete = sum(1 for record in records if len(gold._submitted(record)) >= 2)
    print(f"saved {len(records)} overlap records ({complete} answered by two people) to {path}")
    return 0


def judge_manifest_path(name: str, folder: Path | str = "out") -> Path:
    return Path(folder) / f"{name}-judge.manifest.json"


def load_judge_manifest(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def cmd_export(args) -> int:
    from . import argilla_io

    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    questions = load_questions(args.questions)
    items = cand.read_candidates(args.candidates)
    reviews = argilla_io.pull(name=args.name, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_suffix(".reviews.json").write_text(json.dumps(reviews, indent=2, default=str))
    # Every export also snapshots the overlap: the first export, which lists the disagreements, is taken
    # before they are resolved, so later adjudication cannot raise the reported kappa.
    folder = Path(args.out).parent / "snapshots"
    earlier = load_overlap_snapshots(folder, args.name)
    if reviews.get("overlap"):
        save_overlap_snapshot(folder, args.name, reviews, taken_by="export")
    manifest = load_judge_manifest(judge_manifest_path(args.name, Path(args.out).parent))
    return export_from_reviews(args, corpus, questions, items, reviews, snapshots=earlier, judge_manifest=manifest)


def export_from_reviews(args, corpus, questions, items, reviews, snapshots: list[dict] | None = None,
                        judge_manifest: dict | None = None) -> int:
    from . import screening

    ontology = ontology_hash()
    report = gold.decide(corpus, {item.candidate_id: item for item in items}, reviews, args.adjudicator)
    added, missing_problems = gold.missing_claims(corpus, reviews)
    rows = gold.review_rows(questions, items, report, added, corpus=corpus, version=args.version,
                            ontology_hash=ontology)
    strata_path = Path(getattr(args, "candidates", None) or "out/candidates.jsonl").with_suffix(".strata.json")
    strata = json.loads(strata_path.read_text()) if strata_path.exists() else {}
    cand.write_jsonl(args.out, rows)
    warnings: list[str] = []
    overlap, source = gold.overlap_from_snapshots(snapshots or [], reviews.get("overlap", []))
    agreement = gold.agreement({"overlap": overlap})
    if source["records_from_live"]:
        warnings.append(
            f"agreement: {source['records_from_live']} overlap records are scored from responses that are in no "
            "earlier snapshot; if their disagreements were already resolved by editing a response, kappa is "
            "inflated. Run `labelling snapshot` after the independent round and before adjudicating.")
    adjudicator = args.adjudicator
    raters = set(agreement.get("reviewers") or [])
    adjudication = {"adjudicator": adjudicator, "resolved": report.adjudicated,
                    "resolved_by_a_rater": report.adjudicated_by_a_rater,
                    "adjudicator_is_overlap_rater": bool(adjudicator and adjudicator in raters)}
    if adjudicator and (adjudication["adjudicator_is_overlap_rater"] or report.adjudicated_by_a_rater):
        adjudication["warning"] = (
            f"the adjudicator {adjudicator!r} is one of the two overlap raters: on {report.adjudicated_by_a_rater} "
            "disagreements their own answer won. Adjudicate with a third person, or record a discussion and "
            "enter its outcome under a separate user (guidelines/annotation.md, section 6).")
        warnings.append("adjudication: " + adjudication["warning"])
    judge = gold.judge_agreement(reviews, judge_manifest) if reviews.get("judge") else None
    summary = {
        "gold_rows": len(rows),
        "kept": sum(d.keep for d in report.decisions),
        "rejected": sum(not d.keep for d in report.decisions),
        "unreviewed": report.unreviewed,
        "disagreements": len(report.disagreements),
        "unresolved_disagreements": sum("resolved_by" not in row for row in report.disagreements),
        "adjudication": adjudication,
        "agreement": agreement,
        "agreement_source": source,
        "agreement_live": gold.agreement(reviews),
        "judge_agreement": judge,
        "screening_check": screening.report(reviews) if reviews.get("screening") else None,
        "snapshot": gold.snapshot(),
        "problems": report.problems + missing_problems,
        "triage": triage_miss(items, report, strata),
        "warnings": warnings,
    }
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(
        {**summary, "disagreement_details": report.disagreements}, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if summary["snapshot"]["corpusSnapshotId"] == "unknown":
        print("warning: CORPUS_SNAPSHOT_ID is unset; set it (same value) in this .env and the harness .env")
    return 0


# ─── The screening check (papers D marked not relevant) ─────────────────────

def cmd_screening_sample(args) -> int:
    from . import screening

    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    only = [part.strip() for part in (args.only or "").split(",") if part.strip()] or None
    size = doc_settings().screening_sample if args.sample is None else args.sample
    records, strata = screening.sample_records(corpus, docs_root(args.docs), only, size=size, seed=args.seed)
    cand.write_jsonl(args.out, records)
    Path(args.out).with_suffix(".strata.json").write_text(json.dumps(strata, indent=2))
    for key, sizes in sorted(strata.items()):
        print(f"{key}: {sizes['sampled']} of {sizes['not_relevant']} papers marked not relevant"
              + (f" ({sizes['unread']} unread, not in the frame)" if sizes["unread"] else ""))
    print(f"wrote {len(records)} screening records to {args.out}; push with: labelling screening-push")
    return 0


def cmd_screening_push(args) -> int:
    from . import argilla_io

    rows = read_jsonl(Path(args.sample_file))
    if not rows:
        raise SystemExit(f"no screening records in {args.sample_file}: run `labelling screening-sample` first")
    count = argilla_io.push_screening(rows, name=args.name, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"))
    print(f"pushed {count} papers to {args.name}-screening")
    return 0


def cmd_screening_report(args) -> int:
    from . import argilla_io, screening

    reviews = argilla_io.pull(name=args.name, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"),
                              suffixes=("screening",))
    strata_path = Path(args.sample_file).with_suffix(".strata.json")
    strata = json.loads(strata_path.read_text()) if strata_path.exists() else None
    result = screening.report(reviews, strata)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


def triage_miss(items, report, strata) -> dict:
    from . import triage

    return triage.miss_report(items, report.decisions, strata)


def judge_cache_path() -> Path:
    """The harness judge cache: HARNESS_JUDGE_CACHE_FILE, else $HARNESS_ARTIFACTS/judge_cache.jsonl."""
    configured = os.getenv("HARNESS_JUDGE_CACHE_FILE", "").strip()
    if configured:
        return Path(os.path.expanduser(configured))
    return env_path("HARNESS_ARTIFACTS", "../rambodambo/artifacts") / "judge_cache.jsonl"


def judge_sample(rows: list[dict], size: int, seed: str) -> list[dict]:
    """A deterministic sample stratified by the judge's answer: half "same fact" and half not when both are
    available. The strata are not in their population proportions, so ``judge-push`` records the
    population sizes and ``judge-agreement`` reweights by them."""
    ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{seed}:{row['key']}".encode()).hexdigest())
    yes = [row for row in ranked if row["same_fact"]]
    no = [row for row in ranked if not row["same_fact"]]
    half = size // 2
    take_yes = min(len(yes), max(half, size - len(no)))
    return yes[:take_yes] + no[: size - take_yes]


def cmd_judge_push(args) -> int:
    from . import argilla_io

    path = Path(args.cache) if args.cache else judge_cache_path()
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"no judge decisions in {path}; run `make evaluate ARGS=--judge` in the harness first")
    # Only the current prompt version: older decisions are not what the harness uses now.
    prompt = args.prompt or rows[-1].get("prompt")
    rows = [row for row in rows if row.get("prompt") == prompt]
    print(f"judge prompt {prompt}: {len(rows)} decisions")
    chosen = judge_sample(rows, args.sample, args.name)
    count = argilla_io.push_judge(chosen, name=args.name, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"))
    manifest = judge_manifest(rows, chosen, name=args.name, prompt=prompt, cache=path)
    target = Path(args.manifest) if args.manifest else judge_manifest_path(args.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2))
    print(f"pushed {count} of {len(rows)} judge decisions to {args.name}-judge "
          f"({manifest['sample']['yes']} of {manifest['population']['yes']} judged same fact, "
          f"{manifest['sample']['no']} of {manifest['population']['no']} not); manifest: {target}")
    return 0


def judge_manifest(rows: list[dict], chosen: list[dict], *, name: str, prompt: object, cache: Path) -> dict:
    """What judge-agreement needs to undo the stratified sample: population and sample size per stratum."""
    from . import argilla_io

    def label(row: dict) -> str:
        return "yes" if row["same_fact"] else "no"

    return {
        "dataset": f"{name}-judge", "seed": name, "prompt": prompt, "cache": str(cache),
        "pushed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "population": {key: sum(label(row) == key for row in rows) for key in ("yes", "no")},
        "sample": {key: sum(label(row) == key for row in chosen) for key in ("yes", "no")},
        "record_ids": {key: [argilla_io.judge_record_id(row) for row in chosen if label(row) == key]
                       for key in ("yes", "no")},
    }


def cmd_judge_agreement(args) -> int:
    from . import argilla_io

    reviews = argilla_io.pull(name=args.name, workspace=os.getenv("ARGILLA_WORKSPACE", "thesis"),
                              suffixes=("judge",))
    manifest = load_judge_manifest(Path(args.manifest) if args.manifest else judge_manifest_path(args.name))
    result = gold.judge_agreement(reviews, manifest)
    print(json.dumps(result, indent=2))
    for warning in result["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    if result["kappa"] is not None and not result["use_judge"]:
        print(f"{result['kappa_basis']} kappa below {gold.KAPPA_BAR}: report soft scores from reviewer variants "
              "only (see docs/EVALUATION.md)")
    return 0


def cmd_install(args) -> int:
    target = env_path("HARNESS_ROOT", "../rambodambo") / "datasets" / "gold" / "candidates.jsonl"
    gold = Path(args.gold)
    if not gold.exists():
        print(f"no gold file at {gold}: run `labelling export` first (it needs Argilla)", file=sys.stderr)
        return 2
    new_rows = read_jsonl(gold)
    if not new_rows:
        print(f"{gold} has no gold rows; nothing installed", file=sys.stderr)
        return 2
    existing = read_jsonl(target)
    new_ids = {row["id"] for row in new_rows}
    kept = [row for row in existing if row.get("id") not in new_ids]
    if target.exists():
        backup = target.with_suffix(f".{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.bak")
        shutil.copy2(target, backup)
        print(f"backup: {backup}")
    temporary = target.with_suffix(".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    cand.write_jsonl(temporary, kept + new_rows)
    temporary.replace(target)
    print(f"installed {len(new_rows)} gold rows into {target} ({len(kept)} other rows kept)")
    from .triage import is_draft

    drafts = sorted(row.get("question_id", row.get("id")) for row in kept + new_rows if is_draft(row))
    if drafts:
        print(f"note: {len(drafts)} DRAFT rows (triage only, not reviewed) are installed: {', '.join(drafts)}."
              " Scores against them are provisional; install the reviewed export to replace them.")
    return 0


def cmd_evidence_docs(rest: list[str]) -> int:
    """Build evidence documents with the harness packages (labelling.evidence_doc; ``--help`` for options).

    The harness's ``exhaustive`` profile applies unless HARNESS_PROFILE names another (``none``: no
    profile). It has to be chosen before the harness settings load, so the import comes after.
    """
    profile = os.environ.get("HARNESS_PROFILE", "").strip() or "exhaustive"
    if profile == "none":
        os.environ["HARNESS_PROFILE"] = ""
    elif "/" not in profile and not profile.endswith(".env"):          # a profile name: the harness's file
        os.environ["HARNESS_PROFILE"] = str(env_path("HARNESS_ROOT", "../rambodambo") / "profiles" / f"{profile}.env")
    from .evidence_doc import main as build

    return build(rest)


def main(argv: list[str] | None = None) -> int:
    load_env()
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["evidence-docs"]:            # its own options (evidence-docs --help)
        return cmd_evidence_docs(argv[1:])
    parser = argparse.ArgumentParser(prog="labelling", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def with_questions(p):
        p.add_argument("--questions", default="questions/questions.json")
        return p

    with_questions(sub.add_parser("validate")).set_defaults(func=cmd_validate)
    p = sub.add_parser("triage")
    p.add_argument("--docs", default=None, help="evidence documents (default HARNESS_DOC_ROOT or evidence_docs/)")
    p.add_argument("--only", default=None, help="comma-separated question ids (default: every built document)")
    p.add_argument("--batch", type=int, default=20, help="triples per model call")
    p.add_argument("--workers", type=int, default=4, help="parallel model calls")
    p.set_defaults(func=cmd_triage)
    p = with_questions(sub.add_parser("pool"))
    p.add_argument("--docs", default=None, help="evidence documents (default HARNESS_DOC_ROOT or evidence_docs/)")
    p.add_argument("--out", default="out/candidates.jsonl")
    p.add_argument("--tiers", default="answers,context", help="triage labels reviewed in full")
    p.add_argument("--sample", type=int, default=50, help="random triples from the other labels, per question")
    p.add_argument("--seed", default="gold-v1", help="sample seed (same seed, same sample)")
    p.add_argument("--all", action="store_true", help="ignore triage: every triple goes to review")
    p.set_defaults(func=cmd_pool)
    p = with_questions(sub.add_parser("draft"))
    p.add_argument("--docs", default=None, help="evidence documents (default HARNESS_DOC_ROOT or evidence_docs/)")
    p.add_argument("--tiers", default="answers,context", help="triage labels kept in the draft answer key")
    p.add_argument("--version", default="v1", help="the gold version the reviewed export will use")
    p.add_argument("--out", default="out/gold-v1.draft.jsonl")
    p.set_defaults(func=cmd_draft)
    p = with_questions(sub.add_parser("push"))
    p.add_argument("--candidates", default="out/candidates.jsonl")
    p.add_argument("--name", default="gold-v1")
    p.add_argument("--overlap", default="auto", help="auto (20%%), none, or comma-separated question ids")
    p.add_argument("--docs", default=None, help="evidence documents (default HARNESS_DOC_ROOT or evidence_docs/)")
    p.set_defaults(func=cmd_push)
    p = sub.add_parser("screening-sample")
    p.add_argument("--docs", default=None, help="evidence documents (default HARNESS_DOC_ROOT or evidence_docs/)")
    p.add_argument("--only", default=None, help="comma-separated question ids (default: every built document)")
    p.add_argument("--sample", type=int, default=None,
                   help="papers per question and interpretation (default HARNESS_DOC_SCREENING_SAMPLE, 5)")
    p.add_argument("--seed", default="gold-v1", help="sample seed (same seed, same sample)")
    p.add_argument("--out", default="out/screening.jsonl")
    p.set_defaults(func=cmd_screening_sample)
    p = sub.add_parser("screening-push")
    p.add_argument("--sample-file", default="out/screening.jsonl")
    p.add_argument("--name", default="gold-v1")
    p.set_defaults(func=cmd_screening_push)
    p = sub.add_parser("screening-report")
    p.add_argument("--sample-file", default="out/screening.jsonl", help="its .strata.json gives N where needed")
    p.add_argument("--name", default="gold-v1")
    p.add_argument("--out", default="out/screening.report.json")
    p.set_defaults(func=cmd_screening_report)
    p = sub.add_parser("snapshot")
    p.add_argument("--name", default="gold-v1")
    p.add_argument("--dir", default="out/snapshots")
    p.set_defaults(func=cmd_snapshot)
    p = with_questions(sub.add_parser("export"))
    p.add_argument("--candidates", default="out/candidates.jsonl")
    p.add_argument("--name", default="gold-v1")
    p.add_argument("--version", default="v1")
    p.add_argument("--adjudicator", default=None,
                   help="username whose response resolves disagreements (a third person, not one of the two raters)")
    p.add_argument("--out", default="out/gold-v1.jsonl")
    p.set_defaults(func=cmd_export)
    p = sub.add_parser("install")
    p.add_argument("--gold", default="out/gold-v1.jsonl")
    p.set_defaults(func=cmd_install)
    p = sub.add_parser("judge-push")
    p.add_argument("--cache", default=None,
                   help="harness judge cache (default: HARNESS_JUDGE_CACHE_FILE or $HARNESS_ARTIFACTS/judge_cache.jsonl)")
    p.add_argument("--prompt", default=None, help="prompt version to sample (default: the latest in the cache)")
    p.add_argument("--sample", type=int, default=60)
    p.add_argument("--name", default="gold-v1")
    p.add_argument("--manifest", default=None, help="where to record the sample (default out/<name>-judge.manifest.json)")
    p.set_defaults(func=cmd_judge_push)
    p = sub.add_parser("judge-agreement")
    p.add_argument("--name", default="gold-v1")
    p.add_argument("--manifest", default=None, help="the judge-push manifest (default out/<name>-judge.manifest.json)")
    p.set_defaults(func=cmd_judge_agreement)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
