# labelling

Gold-set construction for the [Research Harness](https://github.com/guptaaditya746/rambodambo), so
precision and recall measure correctness rather than agreement with another system.

## Principles

- **Two kinds of question**, at least 10 each (`questions/questions.json`): **corpus-wide** questions
  and **ambiguous** ones (professor style), which the agent interprets.
- **One evidence document D per question**, built once (`labelling evidence-docs`): every paper is
  screened and absence is recorded per paper; extraction is **question-scoped**: in each relevant paper,
  the passages the reader cites plus the top 12 by the harness ranker (`HARNESS_DOC_EXTRACT_PASSAGES`;
  `0` extracts every passage, the option for a final build when the model budget allows). An ambiguous
  question gets one section per interpretation. See **[docs/EVIDENCE_DOCS.md](docs/EVIDENCE_DOCS.md)**.
- **Reviewed once, by people**, in Argilla, **stratified**: a model sorts D's triples (triage), people
  judge every triple it put in answers or context plus a random sample of the rest, each against its
  passage, and export estimates what the sort missed. What is correct and answers the question (or
  interpretation) is the **answer key**, the gold. Reviewers add facts D missed, and check a random
  sample of the papers D marked not relevant (`labelling screening-sample`).
- **Independent of the runs it scores**: D is not pooled from the paths being compared (Pipeline,
  Agentic, External). It uses the harness only as a tool, from a checkout pinned at a tag; it shares
  that tool's reader, ranker and extractor, a bias the screening check and the missing-facts step bound
  (docs/EVALUATION.md, section 8).
- **Agreement**: about 20% of the questions are reviewed independently by two people (you and your
  professor); Cohen's kappa (with a bootstrap interval over questions) is computed from a snapshot taken
  before adjudication, and disagreements are adjudicated, preferably by a third person.
- **Compatibility**: gold rows use the harness's identifiers, predicates and `build_claims`, so the
  harness evaluator scores them directly (`tests/test_harness_compatibility.py`).
- **Metrics**: triple precision and recall at three matching levels (strict, entailed, soft), vital
  recall, judged precision, by question complexity, decided before any run:
  **[docs/EVALUATION.md](docs/EVALUATION.md)**.

## Where things run

| What | Where |
|---|---|
| Argilla + Elasticsearch + PostgreSQL + Redis | Omen (Docker), see [infra/README.md](infra/README.md) |
| These scripts, the harness and its models | Slurm compute node |
| Review in the browser | laptop, via `omen-tunnels` (`argilla` service) |

## Setup (compute node)

```sh
git clone https://github.com/guptaaditya746/labelling.git && cd labelling
cp .env.example .env          # paths, model endpoint, Argilla URL/key
make setup                    # uv env with argilla, pytest
git -C $HARNESS_ROOT checkout <tag>   # the harness version whose tools build D
make harness                  # its packages (build_claims, scorer, readers, extraction)
make test
```

**Snapshot.** Gold rows carry the evaluation snapshot the harness stamps on benchmark runs; both must
be identical or the Benchmark page will not pair them. Run `labelling validate`, copy the printed
corpus fingerprint, and set `CORPUS_SNAPSHOT_ID=<fingerprint>` (plus the same `ONTOLOGY_VERSION` and
`RETRIEVAL_INDEX_VERSION`) in **both** this `.env` and the harness `.env`.

## Workflow

| Step | Command | Output |
|---|---|---|
| 1. Questions: rate complexity (two raters), set origin ([rules](questions/README.md)) | `labelling validate` | problems, level and origin counts, fingerprint |
| 2. Build the evidence documents (try `ARGS="--only nlp33 --max-papers 20"` first) | `make evidence-docs` | `evidence_docs/<id>/doc.json`, `doc.md` (resumable) |
| 3. Sort each triple: answers / context / irrelevant (a model; orders review, never gold) | `make triage ARGS="--only nlp33"` | `evidence_docs/<id>/triage.jsonl` (resumable) |
| 4. Review records: every answers + context triple, and 50 random others per question | `labelling pool` (`--sample N`, `--all` = every triple) | `out/candidates.jsonl`, `.strata.json` |
| 5. Load into Argilla | `labelling push` | datasets `gold-v1-review`, `-overlap`, `-missing` |
| 5b. Sample papers D marked not relevant (`--sample N`, default `HARNESS_DOC_SCREENING_SAMPLE`=5 per question and interpretation) and load them | `labelling screening-sample`, `labelling screening-push` | `out/screening.jsonl`, `.strata.json`; dataset `gold-v1-screening` |
| 6. Review ([guideline](guidelines/annotation.md)): you do review + missing + screening; both of you do overlap | Argilla UI | responses |
| 6b. When both have finished the overlap, before discussing it | `labelling snapshot` | `out/snapshots/gold-v1-overlap-<time>.json` |
| 7. Export the answer keys, agreement, what triage and screening missed; then adjudicate the listed disagreements (a third person) and export again | `labelling export --adjudicator <third person>` | `out/gold-v1.jsonl`, `.report.json` (every export also saves an overlap snapshot) |
| 7b. The screening's false-negative rate alone | `labelling screening-report` | `out/screening.report.json` |
| 8. Install into the harness | `labelling install` | `datasets/gold/candidates.jsonl` (backup kept) |
| 9. Evaluation runs: each question 3× per path | in the harness (Benchmark → Run) | runs in `artifacts/` |
| 10. Score, and fill the judge cache | harness `make evaluate ARGS="--judge"` | strict / entailed / soft by complexity × path |
| 11. Check the judge against people | `labelling judge-push`, review in Argilla, `labelling judge-agreement` | `out/gold-v1-judge.manifest.json` (population per stratum); population-weighted kappa and the judge's yes-precision; weighted kappa below 0.61: report soft without the judge |

**Before the review is done:** `labelling draft` writes a DRAFT answer key from triage alone (every
answers + context triple kept, nothing checked), and `labelling install --gold out/gold-v1.draft.jsonl`
installs it, so steps 9–10 can be tried end to end. Its rows carry `gold_version` `v1-draft` and reviewer
`triage-draft`; scores against them are provisional and are not reported. Installing the reviewed export
(step 8) replaces them, since the ids are the same; `install` lists any draft rows still installed.

Gold is built once (steps 1–8) and frozen; every later harness change only repeats steps 9–10.

## What the numbers mean

Short version (details, reasons and limits in [docs/EVALUATION.md](docs/EVALUATION.md)):

- Recall is measured against the **reviewed evidence document** plus reviewer-added facts: recall
  relative to that set, not to everything true in the corpus.
- Strict is the lower bound (no model involved); soft is closest to a person's judgement. Report all
  three; the gap between them is wording, not error.
- An ambiguous question is scored AmbigQA-style: against the union of its reviewed interpretations,
  so answering one of two readings recovers about half; the report also gives recall per interpretation
  (docs/EVALUATION.md section 3.7).
