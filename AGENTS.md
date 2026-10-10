# Repository guide for coding agents (Codex, Claude) and people

`labelling` builds the gold set the Research Harness ([rambodambo](https://github.com/guptaaditya746/rambodambo)) is scored against: one evidence document (D) per question (question-scoped: the reader's passages plus the top-k per paper by the harness ranker; `HARNESS_DOC_EXTRACT_PASSAGES=0` extracts every passage), built with the harness packages as tools, and reviewed once by people in Argilla. Read `README.md` for the workflow and `guidelines/annotation.md` for the review rules.

## Map: where things are

| Task | Look in |
|---|---|
| CLI commands (validate, evidence-docs, triage, pool, push, snapshot, export, install, judge-push, judge-agreement, screening-sample, screening-push, screening-report) | `src/labelling/cli.py` |
| Screening check of papers marked not relevant | `src/labelling/screening.py` |
| Evidence documents (D): the builder, its settings and progress | `src/labelling/evidence_doc.py`, `src/labelling/doc_settings.py`, `docs/EVIDENCE_DOCS.md` |
| What the metrics mean, why they were chosen, their flaws and references | `docs/EVALUATION.md` |
| The question set and complexity rule | `questions/questions.json`, `questions/README.md`, `src/labelling/questions.py` |
| Review records from D (one per triple and interpretation) | `src/labelling/candidates.py` |
| Triage (model sorts D's triples), stratified sample, miss-rate estimate, draft answer key | `src/labelling/triage.py`, `src/labelling/prompts/triage.md` |
| Argilla datasets (review, overlap, missing, judge) | `src/labelling/argilla_io.py`, `infra/argilla/` |
| Export to harness gold format (weight, variants, verdicts, readings, complexity), Cohen's kappa | `src/labelling/gold.py` |
| Corpus loading and fingerprint | `src/labelling/corpus.py` |
| Compatibility with the harness evaluator | `tests/test_harness_compatibility.py` |

## Commands

- `make setup` then `make harness` (installs the harness packages it needs from `HARNESS_ROOT`, checked out at a tag).
- `make test` — ruff and pytest. Run before every commit.
- Workflow commands: `make validate`, `make evidence-docs ARGS="--only nlp33"`, `make triage ARGS="--only nlp33"`, `make pool`, `make push`, `make export ADJUDICATOR=professor`, `make install`.

## Conventions

- Gold rows use the harness's identifiers, predicates and `harness_extraction.claims.build_claims`; the harness evaluator must score them unchanged (the compatibility test checks this).
- `CORPUS_SNAPSHOT_ID`, `ONTOLOGY_VERSION` and `RETRIEVAL_INDEX_VERSION` must match the harness `.env`, or the Benchmark page will not pair gold rows with runs.
- D is built from the harness as a tool, never from the runs it scores; record the harness tag used.

## Do not

- Commit `.env`, Argilla keys or tokens.
- Edit the harness from this repository; change harness packages in rambodambo and reinstall with `make harness`.
- Hand-edit exported gold files; re-run `make export`.
- Report scores against draft gold rows (`gold_version` ending `-draft`); they are for trying the pipeline.
- Show reviewers the triage label or stratum, or keep a triple as gold because triage called it an answer.
- Hand-edit a built `doc.json` instead of reviewing it in Argilla (review decisions must be recorded).
