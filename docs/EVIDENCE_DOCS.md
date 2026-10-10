# Evidence documents (D)

## The idea

For each evaluation question there is one **evidence document D**: everything the corpus says about
the question, written in a fixed vocabulary, built once, reviewed once, and then treated as gold.
Every research path of the harness (Pipeline, Agentic, External) is scored against it. Only two kinds
of question are used (the harness's `datasets/question_set_d.json`):

| Category | Questions | Interpretations in D |
|---|---|---|
| `corpus_wide` | nlp32–nlp47 (16) | one: the question as asked |
| `ambiguous` | nlp30, nlp48–nlp53, nlp62–nlp66 (12) | the agent's 3–4 concept readings, each with its own section |

The builder lives here, apart from the system it measures (decision 0003 in the harness). It uses the
harness packages as tools (paper readers, concept readings, schema inference, open extraction), installed
from a checkout pinned at a tag; `doc.json` records that checkout's commit (`built.harness_commit`).

## What a document contains

`evidence_docs/<question id>/doc.json` (and `doc.md` to read, `<interpretation>.subschema.ttl`),
per interpretation:

| Section | Question it answers |
|---|---|
| `schema` | Which classes and properties may facts use? Seed terms (DBLP, CS-KG core) and invented ones, with origin. |
| `coverage` | Which papers were read, and what was found? One row per indexed paper: `relevant`, `not_relevant`, `no_text`, or `error` (retried on the next build). This is what makes D exhaustive over papers. |
| `triples` | The facts of the extracted passages: `s`, `p`, `o`, the paper, and every passage that states it with its evidence. |
| `claims` | Claim subgraphs built from the triples; each points to its triple ids and adds no facts. |
| `answer_key` | Which triples and claims answer this interpretation. Starts as `pending_review`; filled in review. |

Scoring then uses three numbers per run: grounding precision (run triples that are in D), answer
precision and answer recall (against the answer key).

## Building

```sh
git -C $HARNESS_ROOT checkout <tag>      # the harness version whose tools build D
make harness                              # its packages into this environment
make evidence-docs                        # every question of HARNESS_DOC_QUESTION_SET
make evidence-docs ARGS="--only nlp33,nlp48"
```

For each interpretation:

1. **Schema**: schema inference over the `HARNESS_DOC_SCHEMA_PASSAGES` highest-ranked passages.
2. **Screening, every paper**: a reader sees the paper's `HARNESS_DOC_SCREEN_PASSAGES` highest-ranked
   passages for the question (or its first passages when none rank) and reports findings or nothing.
3. **Extraction, the question's part of each relevant paper** (question-scoped, not every passage): the
   passages the reader cited and their neighbours, plus the paper's `HARNESS_DOC_EXTRACT_PASSAGES` (12)
   highest-ranked passages for the question by the harness ranker (`hybrid_rank`); open extraction,
   definition and mapping as in a research run, with the harness's graph settings.
   `HARNESS_DOC_EXTRACT_PASSAGES=0` extracts every passage of each relevant paper instead: about 26,000
   triples for nlp33, mostly facts that do not concern the question, and extraction calls grow from the
   scoped passages of a paper to all of them (a paper can have a hundred). It is the option for a final
   gold build when the model budget allows; the default keeps cost down.
4. **Claims and files.** The answer key is left for review.

**Ranker bias.** Both the passages a screening reader sees and the default extraction scope are chosen by
the harness's own ranker, which the harness paths also use to find evidence. A fact in a passage the
ranker scores low is less likely to reach D, and so the gold, which favours the harness paths over the
External one (docs/EVALUATION.md, section 8, flaw 1). `doc.json` records what was in force:
`built.screening_scope` and `built.extraction_scope` in words, `built.settings.doc_screen_passages` and
`doc_extract_passages` as numbers. The screening check (below) and the missing-facts records bound what
this misses; `HARNESS_DOC_EXTRACT_PASSAGES=0` removes the extraction half of it.

The harness's `exhaustive` profile applies (broad schema, dense linking, unmapped relations kept, one
passage per extraction call) unless `HARNESS_PROFILE` names another (`none` for none). The
`HARNESS_DOC_*` settings are in `.env.example` and `src/labelling/doc_settings.py`; by default every
paper is screened and extraction is question-scoped (step 3).

**Watching a build.** In a terminal each stage shows a bar: papers screened (with the number found
relevant) and then passages extracted (with the triples found), the rate, the time left, and the model
calls finished, failed and in flight. Without a terminal (or with `--no-bar`) a line is printed about
every 5%. While a single long call runs (interpretations, schema inference) a line every 20 s says
what it is waiting on. `progress.json` in the question's folder holds the same numbers:

```sh
make evidence-docs ARGS="--status"                  # one line per question
watch -n 10 cat evidence_docs/nlp33/progress.json
```

For a quick trial, `ARGS="--only nlp33 --max-papers 20"` screens only the 20 papers that rank highest for
the question (the cache keeps them, so the full build continues from there).

**Changing the scope later.** Every extracted triple keeps its passage, so lowering
`HARNESS_DOC_EXTRACT_PASSAGES` and building again cuts an earlier, larger build to the new scope without
model calls; `counts.triples_outside_scope` (and per paper in the coverage rows) says how many triples
fell outside. Raising it (or setting `0`, every passage) and building again extracts only the passages
not extracted yet: the cache records each paper's extracted passage ids (`passage_ids`), and screening
is not repeated. Cache entries written before those ids were recorded count as complete for a scope no
larger than theirs; for a wider one, the passages their triples do not come from are extracted (again).
A failed extraction batch is left out of `passage_ids` and retried on the next build.

**Resuming.** Progress is appended to `cache.jsonl` in the question's folder after every paper, so a
stopped build resumes where it stopped and a finished one is rebuilt without model calls. A build
started in the harness (before the builder moved here) resumes with
`ARGS="--only nlp33 --out <harness>/datasets/evidence_docs"`, or by copying its folder into
`evidence_docs/`. Delete a folder to start that question again. `HARNESS_DOC_CALL_TIMEOUT_S` bounds
each call and `HARNESS_DOC_CONCURRENCY` sets how many papers are processed at once (the main lever on
time, as far as the model server keeps up). Cost grows with questions × interpretations × papers.

`doc.json`, `doc.md`, `triage.jsonl` and the `.subschema.ttl` files are committed once reviewed; `cache.jsonl` and
`progress.json` are not.

## Triage and stratified review

A document has about 1,000 triples per 20 relevant papers, too many to review one by one. So review is
stratified (`labelling triage`, then `labelling pool`):

1. **Triage.** A model (`TRIAGE_MODEL`, default the harness's high model) sorts every triple, given the
   question (or interpretation) and its passage, into **answers**, **context** (needed to read an answer:
   metric, setting, comparator) or **irrelevant**. When unsure it chooses context. Labels go to
   `<id>/triage.jsonl`; a failed batch is retried by running the command again.
2. **Pool.** Every answers and context triple is reviewed, plus a random sample (`--sample`, default 50
   per question, fixed by `--seed`) of the irrelevant ones. Unsorted triples are reviewed in full. The
   records are shuffled, and reviewers never see the triage label or the stratum.
3. **Export** reports, per question, the kept share of the sample and from it the estimated number of
   gold triples triage put outside the reviewed tiers (with a 95% Wilson interval) and the estimated
   recall of the reviewed tiers with its interval (`triage` in `.report.json`): `estimated_recall_95` =
   tiers_kept / (tiers_kept + sample_kept + N x [high, low]), with N the unreviewed rest and [low, high]
   the Wilson interval of the sample's kept share. A sample with no kept triple gives a point estimate of
   1.0 but not an interval of 1.0. Only decided records count: tier records still open or in an
   unresolved disagreement (`tiers_unresolved`) are missing from tiers_kept and bias the recall
   estimate down; open sample records (`sample_unresolved`) shrink the sample. The sample's kept triples
   are gold like any other; the estimate says how many more were never reviewed.

Triage only orders the work: no triple becomes gold without a person keeping it. The one exception is
on purpose and temporary: `labelling draft` keeps every answers and context triple as a **draft** answer
key, so the harness can be scored end to end before the review. Draft rows say so (`gold_version`
`<version>-draft`, reviewer `triage-draft`, a review note), and the reviewed export replaces them. If the estimated
misses are large, lower the bar (`--tiers answers,context,irrelevant` reviews everything) or raise the
sample.

## Screening check

A paper the screening marked `not_relevant` is never extracted, so a wrong "not relevant" drops its facts
from the gold without trace. `labelling screening-sample` draws a seeded random sample of those papers
per question and interpretation (`--sample`, default `HARNESS_DOC_SCREENING_SAMPLE`=5; `--seed`), writes
`out/screening.jsonl` and `.strata.json` (N marked not relevant, papers drawn, unread papers), and
`labelling screening-push` loads it into the `<name>-screening` dataset. Each record shows the question
and every passage of the paper and asks: "Is this paper relevant to the question? If yes, which facts are
missing?" (guidelines/annotation.md, section 5). `labelling screening-report` (and every export, under
`screening_check`) estimates per section the screening's false-negative rate with a 95% Wilson interval,
and the relevant papers among all N marked not relevant (rate x N, interval x N); a paper counts as
relevant when anyone who checked it says so. Papers left unread (`no_text`, `error`) are outside the
frame and counted separately. Facts added on these records join the gold like missing-facts lines. A
high rate means the screening reader or its passage budget (`HARNESS_DOC_SCREEN_PASSAGES`) is too
narrow: raise it and rebuild, rather than patching the gold paper by paper.

## Review

Review each `doc.json` once: check the screening sample (above), accept or fix
triples, add missing ones, merge relation names that mean the same (one name stays; the others become
rules in the harness's `datasets/ontology/predicate_relations.ttl`), and fill the answer key. Record the
reviewed version with `CORPUS_SNAPSHOT_ID` and `ONTOLOGY_VERSION`; a change to either means rebuilding.
