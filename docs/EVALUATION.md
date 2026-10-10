# Evaluation: metrics, how to read them, their flaws, and why we chose them

The Research Harness answers a question with an answer graph: a set of triples
(subject, relation, object) backed by evidence passages. We evaluate that graph against a **gold set of
triples** built in this repository, and we group the results by **question complexity**, decided before
any run. This document defines every number, says how to read it, lists what it cannot tell you, and
records the reasoning and literature behind each choice.

The implementation lives in the harness (`packages/evaluation`); this repository builds the gold and
holds this document, so that the metric definitions sit next to the data they are defined on.

## Contents

1. [What is evaluated](#1-what-is-evaluated)
2. [How the gold is built](#2-how-the-gold-is-built)
3. [Triple precision and recall](#3-triple-precision-and-recall)
4. [Question complexity](#4-question-complexity)
5. [Aggregation and confidence intervals](#5-aggregation-and-confidence-intervals)
6. [Worked example: nlp27](#6-worked-example-nlp27)
7. [How to read the numbers](#7-how-to-read-the-numbers)
8. [Known flaws and limits](#8-known-flaws-and-limits)
9. [Decisions and the reasoning behind them](#9-decisions-and-the-reasoning-behind-them)
10. [Where it is implemented](#10-where-it-is-implemented)
11. [References](#11-references)

---

## 1. What is evaluated

- **Unit:** one attempt at one question on one path (Pipeline = `baseline-enhanced`, Agentic = `agentic`,
  External = `external`, the mini-SWE-agent baseline): usually one run, whose chosen reading's triples are
  compared with the question's gold triples. An Agentic attempt at an ambiguous question researches each
  proposed reading in its own run; those runs share an attempt id and their chosen readings' triples are
  scored together as one prediction (section 3.7).
- **Main metric:** triple precision, recall and F1, reported at three matching levels (strict, entailed,
  soft), plus **vital recall** and **judged precision**.
- **Grouping:** every score is reported per question, and summarised by **complexity level × path**
  with 95% confidence intervals over questions.
- **Two run inputs, scored apart** (harness `docs/RUN_INPUT.md`): `corpus`, where each path searches the
  papers (finding + extracting + choosing + answering), and `evidence_doc`, where each path gets the
  question's evidence document D and only picks the triples that answer (choosing + answering, from the
  same triples for every path). Comparing a path across the two separates "did not find it" from "found
  it but did not choose it". `make evaluate ARGS="--input evidence_doc"` scores the second.
- **Not evaluated here:** the prose of the answer (fluency, style), latency and cost (reported next to
  the scores on the Benchmark page, but not part of them), and claims outside the question's scope.

Why triples: the harness's product is a knowledge graph, and triple precision/recall is the standard
measure for knowledge-graph construction and relation extraction (Text2KGBench, WebNLG, EDC). It is
also what the professor asked for. Answer-level judgements (nuggets, LLM-as-judge) are folded into the
same metric instead of added as a separate one: a "nugget" here is simply a gold triple, and the
judge is one way of matching triples (section 3.3).

---

## 2. How the gold is built

There are two kinds of question, at least 10 each (`questions/questions.json`): **corpus-wide**
questions, answered as asked, and **ambiguous** ones (professor style), which the agent interprets.
For each question one **evidence document D** is built and reviewed once; the reviewed D is the gold.

1. **Build D** (`labelling evidence-docs`, docs/EVIDENCE_DOCS.md). Exhaustive over papers: a reader
   screens every indexed paper for the question (its top `HARNESS_DOC_SCREEN_PASSAGES`=10 passages by
   the harness ranker). Extraction is **question-scoped**: in each relevant paper, the passages the
   reader cites (with their neighbours) plus its top `HARNESS_DOC_EXTRACT_PASSAGES`=12 passages by the
   harness ranker go through open extraction. `HARNESS_DOC_EXTRACT_PASSAGES=0` extracts every passage of
   each relevant paper, the option for the final gold build when the model budget allows (a wider scope
   later extracts only the passages not done yet); `doc.json` records the scope in force
   (`built.extraction_scope`, `built.settings`). D records a coverage row for every paper (relevant, not
   relevant, unread), so absence is recorded, not assumed. An ambiguous question gets one section per
   interpretation the agent proposes. D uses the harness only as a tool, from a checkout pinned at a tag.
2. **Review** (Argilla, `labelling triage`, `labelling pool`, then `labelling push`). A model first
   sorts D's triples into answers / context / irrelevant; every answers and context triple is a record,
   plus a random sample of the rest, so the miss rate of that sort is measured, not assumed (export
   reports the estimated gold triples outside the reviewed tiers and the recall of the reviewed tiers,
   each with a 95% interval). The sort is
   never gold and is hidden from reviewers (docs/EVIDENCE_DOCS.md, "Triage"). Until the review is
   done, `labelling draft` can install a draft answer key from the sort alone (marked
   `gold_version` `<version>-draft`); scores against it test the pipeline and are never reported. The
   reviewer judges each record against its passage only: correct / partly correct (with a fix) / wrong, and
   whether it answers the question (or the interpretation), supports the answer, or is irrelevant.
   Kept triples get a **weight** (vital or okay) and **accepted variants**. See
   `guidelines/annotation.md`.
3. **Missing facts.** Per question (and per interpretation), the reviewer adds correct, relevant facts
   D does not contain, so recall is not bounded by what the builder found.
4. **Screening check.** A seeded random sample of the papers D marked not relevant, per question and
   interpretation (`labelling screening-sample`, `HARNESS_DOC_SCREENING_SAMPLE`=5 by default), is judged
   by people in the `-screening` dataset ("Is this paper relevant to the question? If yes, which facts
   are missing?"). `labelling screening-report` (and export) estimates the screening's false-negative
   rate with a 95% Wilson interval, scaled to the number of papers marked not relevant; facts added there
   join the gold.
5. **Agreement.** About 20% of the questions are reviewed independently by two people; Cohen's kappa is
   reported for verdict, relevance, keep/drop (with the export's own decision rule) and weight, with a
   95% bootstrap interval over questions. It is computed from the earliest snapshot of the independent
   round (`labelling snapshot`; every export also saves one), so resolving a disagreement by editing a
   response does not raise it. Disagreements are adjudicated, by a third person or a recorded discussion;
   an adjudicator who is one of the two raters is flagged in the export report.
6. **Freeze.** `labelling export` writes gold rows (the answer keys: kept, relevant triples, with a
   version, the corpus fingerprint, the ontology hash and the evaluation snapshot); `labelling install`
   puts them in the harness. The gold is then fixed; later harness changes only rerun and rescore.

What a gold row carries (fields the scorer uses):

| Field | Meaning |
|---|---|
| `gold_triples` | The triples a correct answer must contain, one per normalised (s, p, o) with every passage that states it in `sources`; each may have `weight` and `variants` (empty for an ambiguous question) |
| `accepted_readings` | For an ambiguous question: one entry per reviewed interpretation of D, with its own `gold_triples`; the prediction is scored against the union of all readings' gold, with recall per reading (section 3.7) |
| `forbidden_alternatives` | Reviewer-rejected triples, with their `verdict` and `relevance` |
| `complexity` | `features`, `points`, `level` (section 4) |
| `snapshot`, `corpus_fingerprint`, `ontology_hash`, `gold_version` | Which corpus and ontology the gold is valid for |

**Why a document built per question, and not a pool of system outputs.** Pooling the outputs of the
systems being compared (the TREC method) biases recall towards those systems: a fact none of them found
cannot enter the gold. D is built by screening every paper for the question, independently of the runs
it scores, so no compared path's outputs enter it; people still decide what is correct and what answers
the question. It is not free of that bias, though: D comes from one producer that shares the harness's
reader, ranker and extractor, and its default extraction is question-scoped by that ranker, so it can
favour what the harness paths find (section 8, flaw 1). The builder is a model pipeline, so D is not gold
until reviewed, and the missing-facts step and the screening check (a random sample of "not relevant"
papers judged by people, with an estimated false-negative rate) bound what it can miss.

Not yet in the scorer: **grounding precision** (the share of a run's triples that are correct triples
of D, relevant or not); today a run is scored against the answer key only.

---

## 3. Triple precision and recall

### 3.1 Definitions

For one run and one gold set, after both are **deduplicated by canonical key** and **grouped into
facts** (below):

- **Precision** = run facts credited ÷ run facts scored
- **Recall** = gold facts recovered ÷ gold facts
- **F1** = 2PR / (P + R)

**A fact, not a spelling, is the unit.** Triples that entail each other under the rules of section 3.3
(an inverse restatement `MethodX uses ToolQ` / `ToolQ usedBy MethodX`, a symmetric relation read
backwards, two same-as names) are one fact on each side, at every level; a gold fact is vital if any of
its triples is, and it pools their variants. Euzenat's ratios are taken over these facts. Without this,
a run stating a gold fact in both directions lost precision, a gold listing both directions halved the
recall of a run stating one, and stating everything twice could game a set-based score. With single-
premise rules mutual entailment is an equivalence and its members have the same consequences, so one
representative stands for the fact. The scorer lists the grouped triples (`equivalent`).

Credit is **one-to-one**: a maximum matching between run facts and gold facts, so one gold fact
is never counted twice and one run fact never recovers two facts. This is stricter than Euzenat's
set-based definition, where one run triple that implies two gold triples (a relation and the class its
domain implies, say) recovers both; here it recovers one. It is computed as a weighted
assignment (SciPy's `linear_sum_assignment`) whose weights make, in this order, more pairs, more vital
gold triples and stricter pairs win; so vital recall never drops at a looser level and each pair is
reported at the strictest level available.

**Canonical key** (the strict comparison): case and spacing are normalised; DBLP links are compared
case-insensitively; author names become one slug; paper keys like `p12` and the `hrn:` prefix are
resolved; the harness predicates `title`, `author`/`authoredBy`, `year`, `venue`, `cites` are mapped to
their DBLP IRIs (`dblp:title`, `dblp:authoredBy`, `dblp:yearOfPublication`, `dblp:publishedIn`,
`dblp:cites`).

### 3.2 Locally closed scoring of DBLP triples

A run triple whose predicate is a DBLP predicate is **only scored if the question's gold (or one of
its variants) uses that predicate, or one related to a gold predicate by a rule** (parent, child or
inverse, section 3.3), **or if a triple it implies uses such a predicate**. Otherwise it is listed as
"not scored" and does not count against precision.

- *Example:* for "How do SafeRAG and HydraRAG differ…", the run also emits
  `SafeRAG authoredBy "Xun Liang"`. It is true, but the question does not ask for authors, so it is
  neither a hit nor an error.
- *Why:* the gold lists what the question needs, not everything true about the papers. Counting true
  but unrequested metadata as false positives would punish the harness for its fact path. This is the
  "locally closed" assumption of Text2KGBench (Mihindukulasooriya et al., 2023).
- *Why only DBLP predicates:* DBLP facts are cheap to emit and always true; invented predicates are
  where the system's real claims (and errors) are, so they are always scored. This is a partial
  adoption: Text2KGBench closes every relation of the ground truth, so a fully locally closed precision
  would be higher than ours whenever a run states true facts with relations the gold never uses.

### 3.3 Three matching levels

The levels are **cumulative**: each keeps every match of the level before and adds more. Every run
triple is credited at the strictest level it matches. Reporting several levels side by side follows the
WebNLG text-to-RDF evaluation (strict / exact / partial / type).

| Level | A run triple matches a gold triple when… | Who provides the input | Model? |
|---|---|---|---|
| **Strict** | the canonical keys are equal | nobody (normalisation rules in code) | no |
| **Entailed** | one logically implies the other under RDFS/OWL rules | people, once: `datasets/ontology/predicate_relations.ttl` in the harness | no |
| **Soft** | a rule links them in either direction, the run triple equals one of the gold triple's accepted variants, or the judge says they state the same fact | reviewers (variants, in Argilla); the judge only for leftovers | only the judge |

#### Strict

Exact equality after normalisation. It is the **lower bound**: no model, nothing to dispute. It
undercounts whenever the same fact is worded differently, which is frequent (BenchIE showed that
benchmarks without all acceptable wordings misjudge systems; Gashteovski et al., 2022).

#### Entailed

The rules are ordinary W3C RDFS/OWL statements:

| Rule | Example | Effect |
|---|---|---|
| `rdfs:subPropertyOf` | `hrn:hasAttackTask rdfs:subPropertyOf hrn:evaluatesAttack` | `X hasAttackTask Y` implies `X evaluatesAttack Y` |
| `owl:equivalentProperty` | `hrn:addresses owl:equivalentProperty hrn:addressesProblem` | two names for one relation |
| `owl:inverseOf` | `dblp:authorOf owl:inverseOf dblp:authoredBy` | `A authorOf P` implies `P authoredBy A` |
| `rdfs:subClassOf` | `hrn:HybridRAGMethod rdfs:subClassOf hrn:Method` | `X type HybridRAGMethod` implies `X type Method` |
| `owl:sameAs` | `hrn:HydraRAG owl:sameAs hrn:HydraRAG-framework` | two names for one entity |
| `owl:SymmetricProperty` | `dblp:coAuthorWith` | `A coAuthorWith B` implies `B coAuthorWith A` |
| `rdfs:domain` / `rdfs:range` (RDFS rules rdfs2, rdfs3) | `hrn:uses` has domain and range `ResearchEntity` | `X uses Y` implies `X type ResearchEntity` and `Y type ResearchEntity` (a range is used only for object properties) |

Rules are applied to the raw subject and object, and every derived triple is keyed again, because
entity keys depend on the predicate (an author name is normalised as a person under `dblp:authoredBy`,
as plain text elsewhere). The rule set is the harness schema (which already declares `hrn:authoredBy` ≡ `dblp:authoredBy` and so
on), the DBLP seed schema (which brings DBLP's own hierarchy, e.g. `dblp:authoredBy` ⊑ `dblp:createdBy`
and its inverse `dblp:authorOf`), and the predicate-relations file. The combined graph is closed once
with the OWL 2 RL reasoner (`owlrl`), so chains and equivalences are resolved by a standard reasoner.
Every supported rule has a single premise, so a triple's consequences can be computed on its own (the
union over triples is Euzenat's Cn(R)); that is what lets the scorer link one run triple to one gold
triple. Rules with two premises break this. A property chain (`owl:propertyChainAxiom`) is refused when
the rules are built. Transitive properties are closed per triple only, so `a p b` and `b p c` in a run
do not recover a gold `a p c`; the DBLP seed declares two (`dblp:isVersionOf`,
`dblp:publishedAsPartOf`), bibliographic chains the gold does not ask for, and a test fails if a content
predicate becomes transitive.

**Direction matters** (semantic precision and recall, Euzenat 2007):

- *Recall* credits a gold triple when a run triple **implies** it.
- *Precision* credits a run triple when a gold triple **implies** it.

| Run triple compared with the gold | Precision credit | Recall credit | Why |
|---|---|---|---|
| more **general** (`evaluatesAttack` vs gold `hasAttackTask`) | yes | no | true (the gold implies it), but it does not state the gold fact |
| more **specific** (`hasAttackTask` vs gold `evaluatesAttack`) | no | yes | it states the gold fact, but the gold does not vouch for the extra detail (e.g. gold `createdBy X` does not prove `authoredBy X`: X could be an editor) |

**This is the level that reconciles invented predicates with DBLP-only gold.** When the harness
invents `firstAuthor`, one rule (`hrn:firstAuthor rdfs:subPropertyOf dblp:authoredBy`) lets its
triples recover the DBLP gold, with no model involved.

**Adding rules.** Add a rule only when it holds in every context, and record why in a comment; anything
weaker belongs to the soft level. The file currently holds no invented-predicate rules: the harness
schema and DBLP seed cover the DBLP side, and rules for invented predicates should be written from
predicates that actually appear in reviewed gold.

#### Soft

Soft approximates a person asking "do these two triples state the same fact?". It is **symmetric**: a
rule link in either direction counts both ways, then two more sources are tried in order.

1. **Accepted variants** (no model), partial synsets. The reviewer typed other correct wordings for
   each kept triple, either another object text or a whole triple. A run triple equal (after
   normalisation) to a variant matches. This borrows BenchIE's "fact synset" idea (store the acceptable
   surface forms of a fact) only in part: variants are optional and not exhaustive, so an unlisted
   correct wording falls to the judge or is missed; and reviewers cannot declare two D triples one fact,
   so two differently worded kept triples stay two gold facts unless a rule links them. Copies of the
   same normalised (s, p, o) kept from different passages are merged on export into one gold triple that
   lists every passage (`sources`).
2. **The judge** (a model), only for what the rules leave over. The strict, rule and variant edges are
   matched first. The judge is then asked about pairs with the **same subject** and no rule edge in
   which (a) the run triple is left unmatched, against any gold triple, or (b) the run triple is matched
   but its gold partner also has an edge to a leftover run triple, against a leftover gold triple
   (an augmenting path of length three: the leftover takes the partner's gold and the partner takes
   the leftover gold). Longer paths, which would need a judged edge between two rule-matched triples,
   are not searched. Before, a triple with any rule edge was never judged, even when its rule partner
   was taken by another triple, which biased soft scores down. "Same subject" holds when any triple of
   the run fact and any triple of the gold fact share it, so an inverse restatement is compared from the
   side the gold states it. The judge sees only the two facts (prompt
   `evaluation/triple_match` in the harness) and returns `{"same_fact": true|false, "reason": …}`. It
   is told the subjects are the same entity, that vaguer or different claims are not the same, that
   numbers must agree exactly, and to answer false when unsure.

Safeguards on the judge:

- **Deterministic and cached.** Every decision is stored with the judge model alias and the prompt's
  version hash, and both are part of the cache key: a changed prompt or another judge model
  (`HARNESS_MODEL_JUDGE`) never reuses old answers, and re-scoring with the same settings gives the same
  number. The Benchmark page reads the
  cache only; `make evaluate ARGS="--judge"` asks for unseen pairs. Unasked pairs are reported as
  `judge_pending`, and soft scores are a lower bound until they are filled. A failed or unparseable
  judge call also leaves the pair pending; it never drops a run from the scores.
- **Only where it is reported.** The judge is consulted for the prediction that is scored only (the run's
  chosen reading, or an attempt's union of chosen readings) and its per-reading recall.
- **Validated against people.** `labelling judge-push` loads a sample of the judge's decisions into
  Argilla, stratified by the judge's answer (about half "same fact", half not), and records the
  population size of each stratum (N_yes, N_no) and the sampled records in a manifest
  (`out/<name>-judge.manifest.json`). Reviewers answer the same question without seeing the judge's
  answer. Kappa depends on prevalence, so a kappa on the 50/50 sample is not the kappa of the judge's
  real decisions (whose share of "same fact" is generally not one half); `labelling judge-agreement` therefore
  weights each record by N_h / n_h (n_h: answered records of its stratum) and computes kappa from that
  weighted 2×2 table. It reports the raw-sample and the weighted kappa and decides on the weighted one:
  the bar is **weighted kappa ≥ 0.61** ("substantial", Landis and Koch 1977). It also reports the
  judge's yes-precision P(person yes | judge yes), how often a judged soft credit is right, and
  P(person yes | judge no), how many it withholds, each with a 95% Wilson interval: these two, not
  kappa, say how far the judge inflates or deflates soft scores. The overall numbers use one response
  per record (the alphabetically first user who answered it, a rule blind to the answer); each person's
  own kappa against the judge is listed beside them. A push made before the manifest existed falls back
  to the raw kappa, with a warning; push again to fix it. Below the bar, report soft scores from
  variants only: `make evaluate ARGS="--variants-only"`, or `/benchmark?judge=false`. LLM judges must be
  validated per task because their agreement with people varies widely by task (Bavaresco et al., 2025).
- **Precedent.** Soft matching of triples by embedding similarity (G-BERTScore; Saha et al., 2022) or by
  an LLM judge (KGGen's MINE benchmark, 90.2% agreement with manual scoring on 60 pairs; Mo et al.,
  2025) is established; we prefer human variants first and the judge last because a judge is the least
  transparent source.

### 3.4 Vital recall

Recall over the gold triples marked **vital** (reviewers' judgement: "an answer without this fact misses
the point"). It is computed at every level; the Benchmark page shows the soft one. On the overlap, a
triple both reviewers keep is vital only if both marked it vital; copies of one (s, p, o) kept from
several passages merge into one gold triple that is vital if any copy is (as the scorer treats a fact,
section 3.1).

- *Why:* plain recall treats every fact as equal, so a run can score well by listing many minor facts
  while missing the one that answers the question. The TREC 2024 RAG track scores answers by vital
  nuggets for the same reason (its primary metric counts only fully supported vital nuggets).
- *Variant:* a gold fact is matched or not, so this is the track's **strict** vital score (partial
  support earns nothing, not the 0.5 of its non-strict variant).
- *Reading it:* high soft F1 with low vital recall means "lots of correct detail, but not the point".

### 3.5 Judged precision

The share of run triples **known to be correct**:

- *correct* = run triples with a strict, variant or judge match to gold, or a rule link in the
  precision direction (the gold implies the run triple). A run triple that only implies the gold (a more
  specific claim) is not counted as correct: the gold does not vouch for the extra detail;
- *wrong* = run triples equal to a candidate the reviewers **rejected as wrong** (`forbidden_alternatives`
  with `verdict: wrong`);
- judged precision = correct ÷ (correct + wrong), reported with its **coverage** = (correct + wrong) ÷
  run triples scored.

- *Why:* reference-based precision treats every run triple missing from the gold as an error, but the
  gold is incomplete, so some of those "errors" are true. Judged precision only counts triples someone
  actually judged. EDC reports human-judged precision for exactly this reason when there is no complete
  reference; FActScore defines factual precision as the share of atomic facts supported by a source
  (Min et al., 2023).
- *Reading it:* compare it with soft precision. If judged precision is much higher, the gap is mostly
  true facts the gold lacks; send them to review (the next gold version) rather than calling them errors.
  Low coverage means the number rests on few triples.

### 3.6 DBLP part and invented part

Every level also reports precision and recall separately for triples with DBLP predicates and for
triples with invented predicates. The two behave differently (DBLP facts are nearly always right;
invented ones carry the system's reasoning), and mixing them can hide a weakness in either.

### 3.7 Ambiguous questions

An ambiguous question (the professor's "who is the most successful author?") lists its **accepted
readings**, each with the measure it implies ("most papers", "most cited"). A reading with a
computable operation gets its own gold.

**Scoring: AmbigQA set F1.** The gold of an ambiguous question is the **union** of its accepted readings'
gold (plus the row's own `gold_triples`, the triples reviewed without an interpretation; empty for an
ambiguous question in practice, and never repeated in a reading, so nothing is counted twice). A triple
in several readings is one gold fact, vital if any copy is vital. The prediction is scored **once**
against that union, at every level: a predicted triple is correct if it matches any reading's gold, and
recall is over all readings together. A run that answers one of two equally large readings therefore
gets recall ≈ 1/2; an attempt that answers both gets ≈ 1. The exact metric (`chosen`) uses the same
union, so the two stay comparable (`levels_gold` = `readings_union`; `row` for an unambiguous row).

**Per-reading recall.** Beside the set score, every prediction reports `readings_recall` (soft recall
against each reading's own gold, by label) and `readings_covered` (how many readings it recovered
anything of). The Benchmark averages both over attempts (failed attempts count 0). They show which
interpretations a path covers; they are not a second score, and nothing selects a best reading.

**What is scored: an attempt.** The harness proposes one reading when a question has one sensible
interpretation and several only when it is ambiguous (as AmbigQA first decides whether a question is
ambiguous). In an unattended evaluation, one Agentic attempt researches every proposed reading, each in
its own run; the runs share an attempt id (`Run.attemptId`), and the Benchmark unions their chosen
readings' triples into **one prediction** scored as above. Repetitions are attempts; reading runs are
never averaged as repetitions. Reading runs recorded before attempt ids existed are grouped by the
proposal they came from (`parentRunId`); any other run is an attempt of its own (Fast, Adaptive,
External, or a person's pick of one reading in the UI).

- *Why:* in AmbigQA (Min et al., 2020) a prediction is a **set** of (interpretation, answer) pairs,
  scored by F1 against **all** interpretations in an annotation; the maximum is taken over annotators,
  not over interpretations. Scoring each reading separately and keeping the best would reward answering
  one interpretation as much as answering all of them, the opposite of what an ambiguous question asks. ASQA (Stelmakh et al., 2022) likewise asks a long answer
  to cover every disambiguation. The per-reading recall keeps the information "which interpretation was
  answered" without turning it into the score.
- *Limits:* a reading without computable gold is not in the union, so an answer to it can only count as
  unmatched (it lowers reference precision; judged precision is unaffected unless reviewers rejected the
  triples). The union weighs readings by their number of gold triples: a reading with many facts
  dominates recall; the per-reading recall shows when that matters. The harness's readings are
  matched to the gold through the triples, not through their labels, so a reading worded differently
  from the accepted one is still credited for the facts it recovers.

---

## 4. Question complexity

### 4.1 The rule

Complexity is decided **before any run**, from three facts about what a correct answer needs. The level
is computed by code, never typed by hand.

| Fact | Values | Points |
|---|---|---|
| `papers_read`: papers whose text must be read and combined | 0 (DBLP metadata answers it), 1, 2, 3 (= three or more, or corpus-wide) | +1 if ≥ 1; then +1 more for 2, +2 more for 3 |
| `operators`: operations beyond lookup | count, filter, intersection, comparison, ranking, negation, temporal | +1 each |
| `readings`: accepted interpretations | 1, or ≥ 2 (ambiguous) | +2 if ≥ 2 |

**Level:** 0 points = **simple**, 1 = **compound**, 2 or more = **complex**. The points are kept as a finer
scale for correlation analyses.

| Question | Facts | Points | Level |
|---|---|---|---|
| nlp01 List the authors of FACT-AUDIT. | 0 papers read | 0 | simple |
| nlp07 Count the indexed papers published in 2025. | 0 papers, count | 1 | compound |
| nlp08 Which authors have a paper in both ACL 2025 and EMNLP 2025? | 0 papers, intersection | 1 | compound |
| nlp11 What security risks does SafeRAG test? | 1 paper | 1 | compound |
| nlp27 How do SafeRAG and HydraRAG differ…? | 2 papers, comparison | 3 | complex |
| nlp30 Which RAG approach is best, and by which measure? | corpus-wide, ranking, 2 readings | 6 | complex |

### 4.2 Why this rule

The literature measures question complexity in three ways; we build on the first and validate with the
other two.

1. **From the structure of the answer.** SciQA classifies scholarly questions by the number of triple
   patterns and the shape of the query that answers them (Auer et al., 2023); DBLP-QuAD, built on the
   same DBLP graph as our gold, types questions as single fact, multiple facts, count,
   superlative/comparative, union, disambiguation and others (Banerjee et al., 2023); KQA Pro attaches a
   compositional program to each question (Cao et al., 2022); MuSiQue controls the number of connected
   hops (2–4) and is built so that disconnected reasoning (answering without connecting the hops) pays
   off less: such a model scores about 30 F1 points lower on it than on earlier multi-hop datasets, so it
   is harder to cheat (Trivedi et al., 2022); FRAMES counts the documents a question needs (2–15) and its
   reasoning types (Krishna et al., 2025). Our three facts are these ideas reduced to what varies in a
   paper corpus: how many papers' text, which operations, how many readings.
2. **From outcomes.** Adaptive-RAG labels a question by the cheapest strategy that answers it (no
   retrieval, one step, many steps) and, for a question none of them answers, by its dataset's inductive
   bias (single-hop dataset → one step, multi-hop dataset → many steps) (Jeong et al., 2024); item
   response theory learns each item's difficulty from many systems' results (Lalor et al., 2016;
   Rodriguez et al., 2021).
3. **From human ratings** (stars): subjective, so they need an agreement check.

Why not predict complexity from the question text: Adaptive-RAG's classifier, trained to predict the
label from the question alone, reached 54.5% accuracy on three classes, measured against its own
automatically assigned (silver) labels, not human ones (per class: A, no retrieval, 30.5%; B, one
step, 66.3%; C, many steps, 65.5%). Our gold is a set of triples anyway, so the structure a correct
answer needs is known; deriving complexity from it is more reliable and explainable.

### 4.3 Validation

1. **Agreement.** A second person fills the three facts for all questions independently (about 10
   minutes); report kappa on the level. This turns the design notes' 1–3 stars into a checked rubric.
2. **It must predict performance.** After the evaluation runs, triple F1 should fall as the level
   rises (Spearman correlation between points and per-question soft F1, over all paths), and complex
   questions should more often need the Agentic path to reach a vital recall of, say, 0.75
   (Adaptive-RAG's "cheapest path that succeeds"). If neither holds, revise the rule, not the results.
3. **IRT later.** Item response theory estimates an item's difficulty from the responses of many
   systems; with 3 paths and a few model configurations those estimates would be unstable. Its one cheap
   idea to use now: an item on which the weaker path beats the stronger one on every repetition deserves
   a look at its gold (Rodriguez et al. use negative discriminability to find annotation errors).

**Balance.** The analysis needs a spread of levels (roughly 10 / 10 / 10 for 30 questions), and each
level should contain both the professor's and the student's questions; otherwise complexity and
question origin are confounded and cannot be told apart.

---

## 5. Aggregation and confidence intervals

- **Pairing.** A run is paired with a gold row when the question text and the path match and the run's
  evaluation snapshot equals the row's (same corpus, ontology, claim schema, index version).
- **Every attempt counts.** An attempt that ends without a scorable answer (incomplete, no evidence,
  corpus absent, degraded, or produced by another path: a degraded Agentic run that fell back to
  Adaptive is not Agentic) is scored as an **empty prediction**: precision, recall and F1 0, vital
  recall 0. WebNLG, Text2KGBench and TREC score a missing output this way; averaging only answered runs
  rewards a path for failing on hard questions and drops those questions from the paired comparison.
  The answered-only means stay as a secondary view (`answered_only`, `soft_answered_only`) with their
  **coverage** (answered ÷ attempted). Runs waiting for a reading choice or stopped by a person are not
  attempts. Rows whose `expected_status` is not "answered" keep answered-only scoring.
- **Repetitions first.** Each question is attempted about 3 times per path; repetitions (attempts, not
  the reading runs inside one attempt, which form one prediction; section 3.7) are averaged per question
  first, because they are not independent samples.
- **Macro over questions.** Summaries average per-question scores, so each question counts once
  regardless of how many triples its gold has.
- **Intervals.** 95% BCa (bias-corrected and accelerated) bootstrap over questions (2,000 resamples,
  fixed seed; Efron and Tibshirani, 1993, ch. 14). The plain percentile interval covers less than 95%
  at this sample size; when BCa is degenerate the percentile interval is used (`method`). One question
  has no interval. With about 10 questions per level the intervals are wide; that is the honest answer,
  and the reason to report them.
- **Paired differences.** For each level, the mean per-question difference between Adaptive (or Agentic)
  and Fast on the same questions, with its interval and a two-sided **sign-flip permutation p-value**
  (exact up to 13 questions). A difference whose interval excludes 0 is the evidence for "this path is
  better on this level".
- **Uncertainty of the gold itself** (this repository's reports). Overlap kappa: a 95% percentile
  bootstrap over questions (2,000 resamples, fixed seed), since records of one question are not
  independent; the report also says how many overlap records were left out (answered by one person, or
  by another pair). Proportions estimated from samples (triage's miss rate and the reviewed tiers'
  recall, the screening's false-negative rate, the judge's yes-precision): 95% Wilson intervals, sound
  near 0 and 1 and for small samples. A sample with nothing found still has an interval: 0 of 5 papers
  is compatible with a false-negative rate up to about 0.43.

---

## 6. Worked example: nlp27

"How do SafeRAG and HydraRAG differ in the problems they address in retrieval-augmented generation?"
The paper facts are real (ACL Anthology 2025.acl-long.230 and 2025.emnlp-main.730); the run output is
made up so every case appears once. The harness test `packages/evaluation/tests/test_levels.py`
reproduces these numbers.

**Complexity:** 2 papers read, 1 operator (comparison), 1 reading → 3 points → complex.

**Gold (11 triples):**

| # | Triple | Weight | Variants |
|---|---|---|---|
| G1 | SafeRAG · dblp:title · "SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model" | — | |
| G2 | HydraRAG · dblp:title · "HydraRAG: Structured Cross-Source Enhanced Large Language Model Reasoning" | — | |
| F1 | SafeRAG · addressesProblem · security of RAG | vital | "vulnerability of RAG to manipulated retrieved knowledge" |
| F2 | SafeRAG · contributionType · Benchmark | vital | |
| F3–F6 | SafeRAG · evaluatesAttack · silver noise / soft ad / inter-context conflict / white DoS | okay | |
| F7 | HydraRAG · addressesProblem · multi-hop, multi-entity reasoning across knowledge graphs and text | vital | |
| F8 | HydraRAG · contributionType · Method | vital | |
| F9 | HydraRAG · addressesProblem · verifying evidence across sources | okay | |

**Rule used for this example** (it would be added to the predicate-relations file; the shipped file
has it commented out): `hrn:hasAttackTask rdfs:subPropertyOf hrn:evaluatesAttack`.
**Rejected in review:** `HydraRAG · addressesProblem · defending RAG against prompt injection` (wrong).

**Run (9 triples) and verdicts:**

| # | Run triple | Strict | Entailed | Soft |
|---|---|---|---|---|
| S1 | SafeRAG · title · exact title | G1 | G1 | G1 |
| S2 | HydraRAG · title · exact title | G2 | G2 | G2 |
| S3 | SafeRAG · authoredBy · Xun Liang | not scored (locally closed) | | |
| S4 | SafeRAG · evaluatesAttack · soft ad | F4 | F4 | F4 |
| S5 | SafeRAG · hasAttackTask · silver noise | – | recovers F3 (recall only: more specific) | F3 |
| S6 | SafeRAG · addressesProblem · vulnerability of RAG to manipulated retrieved knowledge | – | – | F1 (variant) |
| S7 | HydraRAG · addressesProblem · reasoning across structured and unstructured sources | – | – | F7 (judge) |
| S8 | SafeRAG · evaluatesComponents · 14 RAG components | – | – | – (true, not in gold) |
| S9 | HydraRAG · addressesProblem · defending RAG against prompt injection | – | – | – (rejected as wrong) |

**Scores** (8 run triples scored, 11 gold):

| Level | Precision | Recall | F1 | Vital recall |
|---|---|---|---|---|
| Strict | 3/8 = 0.38 | 3/11 = 0.27 | 0.32 | 0/4 = 0.00 |
| Entailed | 3/8 = 0.38 | 4/11 = 0.36 | 0.37 | 0/4 = 0.00 |
| Soft | 6/8 = 0.75 | 6/11 = 0.55 | 0.63 | 2/4 = 0.50 |

- **Judged precision:** 5 correct (S1, S2, S4, S6, S7), 1 wrong (S9) → 5/6 = 0.83, coverage 6/8.
  S5 is unjudged: it is more specific than the gold fact, which does not vouch for the detail. S8 is
  unjudged too; if reviewers accept it into the next gold version, it becomes correct.
- **DBLP part:** 2/2 precision and recall. **Invented part (soft):** 4/6 precision, 4/9 recall.
- **Reading:** only one triple (S9) is wrong; most of the strict-to-soft gap is wording. But the vital
  facts F2 and F8 (benchmark vs method) are missing, so the answer never states *how* the two papers
  differ, which is what the question asks. Vital recall shows that; F1 alone would not.

---

## 7. How to read the numbers

| You see | It means | Do |
|---|---|---|
| Strict ≪ soft | The system states gold facts in other words | Nothing is wrong; report both. If the gap is large, check that variants and the judge are validated |
| Entailed > strict | Rules connected invented predicates to the gold | Check the rules that fired are sound |
| Soft F1 fine, vital recall low | Correct details, missing the point | Look at the missed vital facts per question (Interpretation window) |
| Judged precision ≫ soft precision | Many run triples are true but not in the gold | Send true extras to review (gold v2), rescore |
| Low judged-precision coverage | Few run triples were ever judged | Treat judged precision as indicative only |
| `judge_pending` > 0 | Some pairs have no judge decision yet | Soft is a lower bound; run `make evaluate ARGS="--judge"` |
| Overlapping intervals between paths | The data cannot separate the paths on this level | Say so; do not rank them |
| Paired difference interval excludes 0 | A path is better on this level, on the same questions | This is the evidence for a claim like "Agentic helps on complex questions" |
| F1 does not fall with complexity | The complexity rule may not capture difficulty | Revisit the rule (4.3) |

**Recall is relative.** Recall is measured against the reviewed evidence document plus reviewer-added
facts. Report it as recall relative to that set, not as recall of everything true in the corpus.

**Automatic scores are for comparing systems, not grading single questions.** In TREC 2024 RAG,
automatic and manual nugget evaluation agreed strongly when ranking whole systems (Kendall's τ 0.887
with post-edited nuggets, 0.727 against fully manual evaluation) but only weakly per question (τ about
0.3–0.44). Read per-question soft scores as diagnostics; draw conclusions from the aggregates.

---

## 8. Known flaws and limits

1. **Builder bias, ranker bias included.** D is extracted by the harness's own tools (readers, ranker,
   extraction), so its gold can favour facts and wordings those tools produce, which the harness paths
   share and the External harness does not (the bias pooling has, Zobel 1998, with one producer). The
   ranker matters twice: a screening reader sees only a paper's top 10 passages by the harness's
   `hybrid_rank`, and by default only the passages the reader cites plus the top 12 by the same ranker
   are extracted, so a fact in a passage the ranker scores low is less likely to reach the gold, which is
   where a path that ranks differently (External) can be wrongly penalised. Mitigations: every paper is
   screened; the screening check (a random sample of papers marked not relevant, judged by people, with
   an estimated false-negative rate and interval); the missing-facts step; the entailed and soft levels
   for wording; a fixed builder version and the screening and extraction scope recorded in D
   (`built.screening_scope`, `built.extraction_scope`); and `HARNESS_DOC_EXTRACT_PASSAGES=0`, which
   extracts every passage of each relevant paper and removes the extraction half of the ranker bias, at
   a cost that grows with each paper's length.
2. **Incomplete gold → false negatives.** Missing gold facts make true run triples look like errors.
   In relation extraction, re-annotating DocRED's missing facts shifted F1 by about 13 points (Tan et
   al., 2022). Mitigations: judged precision, and the loop that sends true extras to review.
3. **Strict undercounts, and variants are partial synsets.** Exact matching misjudges systems when gold
   lacks acceptable variants (BenchIE). Hence three levels; strict alone should not be the headline.
   Reviewer variants do not close the gap: they are optional and not exhaustive, and two differently
   worded D triples of one fact cannot be grouped by reviewers, so they stay two gold facts (only exact
   (s, p, o) copies from different passages are merged on export).
4. **Entailed is only as good as its rules.** Missing rules undercount; unsound rules overcount. Rules
   are few, reviewed and commented; entailed never guesses.
5. **The judge can be wrong or biased.** LLM judges disagree with people in task-dependent ways
   (Bavaresco et al., 2025) and can favour text in their own style (self-enhancement bias; Zheng et al.,
   2023). The default judge model (`HARNESS_MODEL_JUDGE=heavy-model`) is the same model that produces
   the Heavy path's triples, which is exactly the setting where self-preference can inflate soft
   scores. Mitigations: the judge sees only two short facts, answers false when unsure, is validated
   against people (population-weighted kappa ≥ 0.61, with its yes-precision and interval), and soft can
   be reported without it. Use a different judge model when one is available. The check samples about
   60 decisions, so its kappa and precision intervals are wide.
6. **The judge only compares same-subject pairs.** A run fact whose subject is worded differently from
   the gold's (and not linked by `owl:sameAs` or stated in an inverse form) is never sent to the judge.
   This keeps judge calls few and predictable; it undercounts. Augmenting paths longer than three are
   not searched for judged edges (section 3.3).
7. **One-to-one credit penalises paraphrase.** Restatements the rules recognise are one fact (section
   3.1), but two paraphrases the rules do not link (only a variant or the judge does) earn one credit;
   the second counts against reference precision (but not against judged precision). One run triple
   also recovers at most one gold fact, even when it implies two.
8. **Transitive chains are not closed.** Section 3.3; only DBLP's bibliographic properties are
   transitive today.
9. **Local closure can hide errors.** A wrong DBLP triple with a predicate the gold never uses is not
   scored. Invented predicates are always scored, so this only affects DBLP metadata.
10. **Readings without gold, and unequal readings.** An ambiguous question is scored against the union of
   its accepted readings (section 3.7); a reading without computable gold is not in it, so answering that
   reading cannot be covered, and a reading with many gold facts weighs more in recall than one with few.
11. **Small samples.** About 10 questions per level give wide intervals; per-question automatic scores
    are unreliable (section 7). Conclusions must rest on paired differences with intervals.
12. **The complexity rule is a heuristic.** It is explicit and checkable, but its points are a design
    choice. Section 4.3 is how it is tested.
13. **Triples are not the whole answer.** Triple scores do not measure the prose, its coherence, or
    whether it cites the right passages; the harness's claim scores and answer-support checks cover part
    of that separately.
14. **Pairing by question text and snapshot.** A reworded question or a different snapshot silently
    drops a run from the benchmark; question ids would be safer.
15. **The gold's own completeness is estimated from samples.** What triage left unreviewed and what
    screening marked not relevant by mistake are measured on random samples (about 50 triples and 5
    papers per question and interpretation by default), so the estimates carry wide Wilson intervals.
    Only decided records count: tier records still open or in an unresolved disagreement make the
    reviewed tiers' estimated recall too low. Raise the samples where an interval is too wide to
    support a claim.
16. **Agreement is measured on one subset, and adjudication can favour a rater.** Kappa comes from
    about 20% of the questions and two people; it is computed from the earliest snapshot of their
    independent answers, so edits made while resolving disagreements do not raise it, but a snapshot
    taken after a discussion started cannot tell. An adjudicator who is one of the two raters makes that
    rater's answers win every disagreement; the export flags it, and a third person or a recorded
    discussion is the remedy.

---

## 9. Decisions and the reasoning behind them

| Decision | Alternatives considered | Why this one |
|---|---|---|
| Triple precision/recall as the main metric | Answer-level LLM grading only; nugget recall only | The harness's output is a graph; triple P/R is the field's standard (Text2KGBench, WebNLG, EDC) and the professor's requirement. Nuggets become gold triples with vital/okay weights, so both views are one metric |
| One evidence document per question, every paper screened, reviewed once (the professor's proposal) | Pooling the compared systems' outputs (TREC); writing gold from scratch; model-generated gold | Independent of the runs it scores; complete hand-written gold is infeasible; unreviewed model output measures agreement with a model. It still has one producer's bias (section 8, flaw 1) |
| Question-scoped extraction by default (cited passages + top 12 by the harness ranker); every passage as an option | Every passage of each relevant paper always | Cost: every passage multiplies extraction calls by each paper's length and fills D with facts that do not concern the question (about 26,000 triples for nlp33). The price is ranker bias, stated in flaw 1 and recorded in D; `HARNESS_DOC_EXTRACT_PASSAGES=0` is there for the final build |
| Stratified review (triage tiers in full + a random sample), with the miss rate and recall estimated with Wilson intervals | Review every triple of D | About 1,000 triples per 20 relevant papers; the sample turns "triage missed nothing" into a measured estimate |
| Missing-facts step | Trust D as built | The only way recall can exceed what the builder found |
| Screening check: a random sample of "not relevant" papers judged by people, false-negative rate with a Wilson interval scaled to N | Trust the coverage rows; reviewers browse a few papers of their choice | A paper wrongly marked not relevant drops its facts silently; a random sample gives an estimate, a hand-picked one does not |
| Overlap kappa from the earliest snapshot, with the export's keep rule and a bootstrap over questions; a third-person adjudicator | Kappa from live responses at export; one rater adjudicates | Edits made to settle disagreements would inflate live kappa; kappa and gold must count "keep" the same way; records of one question are not independent; a rater adjudicating their own disagreements makes their answer win |
| Three matching levels, all reported | One exact metric; one soft metric | Strict is indisputable but undercounts; soft is closest to people but depends on a model. The range shows how much is wording (WebNLG reports several levels for the same reason) |
| Entailed via RDFS/OWL rules closed with owlrl | Hand-written matching logic; embedding similarity for predicates | Standard W3C semantics, explainable matches, reuses the harness ontology and DBLP schema (the OWL 2 plan), no model |
| Euzenat's asymmetric entailment | Symmetric "either implies the other" | A more specific claim is not guaranteed by a general gold fact; symmetry would credit unverified detail. Soft is symmetric where a person's judgement is wanted |
| Variants before the judge | Judge only | Variants are human, deterministic and free; the judge is the least transparent source, so it only sees what nothing else resolved |
| Judge cached, cache-only in the UI, validated by kappa | Live judge on every page load | Reproducible numbers, no surprise model load, and a measured agreement with people |
| Judge validated on a sample stratified by its answer, kappa reweighted to the population (N_h/n_h), the 0.61 bar on the weighted kappa, yes-precision reported | Kappa on the balanced sample; a simple random sample | A balanced sample sees enough of both answers, but kappa depends on prevalence, so the decision uses the population-weighted table; yes-precision is what soft-score inflation depends on |
| Locally closed DBLP scoring | Score every triple | The gold lists what the question needs; unrequested true metadata is not an error (Text2KGBench) |
| One-to-one matching over facts | Set overlap; one-to-one over spellings | Prevents double credit (WebNLG aligns candidates to references before scoring); counting facts (triples equal under the rules) keeps restatements from being errors or misses (Euzenat's ratios over facts) |
| Failed attempts score 0 | Average answered runs only | A path must not gain from failing on hard questions; coverage and answered-only scores are reported beside it |
| Ambiguous questions: set F1 against the union of the accepted readings; an Agentic attempt researches every reading and is scored as one prediction | Best soft F1 over the readings scored separately; averaging reading runs as repetitions | AmbigQA scores a set of (interpretation, answer) pairs against all interpretations (max over annotators, not interpretations): answering one of two readings must not score like answering both. Per-reading recall keeps which interpretations were covered |
| Vital recall | Plain recall only | Separates "got the point" from "listed many details" (TREC 2024 RAG vital nuggets) |
| Judged precision with coverage | Reference precision only | The gold is incomplete; judged precision counts only triples someone judged (EDC's human precision, FActScore's factual precision) |
| Complexity from three facts about the gold answer | LLM-rated difficulty; stars only; IRT | Structural complexity is established (SciQA, DBLP-QuAD, MuSiQue, FRAMES), explainable and fixed before runs; text-only prediction is weak (Adaptive-RAG 54.5% against its own silver labels); IRT needs more systems |
| Macro averages, BCa bootstrap intervals, paired differences with a sign-flip test | Micro averages; single numbers; percentile intervals | Each question counts once; honest uncertainty with ~10 questions per level (BCa covers better than percentile at small n); paired comparisons use the same questions |
| No prompt-set (v1 vs v2) comparison for now | Judge prompts on the gold | Decided with the student: keep prompt set v2 and spend the gold on paths and complexity first |

---

## 10. Where it is implemented

| What | Where |
|---|---|
| Evidence documents (D) | this repo: `src/labelling/evidence_doc.py`, `docs/EVIDENCE_DOCS.md` |
| Questions, complexity, review fields, gold rows | this repo: `src/labelling/questions.py`, `candidates.py`, `gold.py`, `argilla_io.py`, `cli.py` |
| Triage, stratified review, miss rate and recall intervals | this repo: `src/labelling/triage.py` |
| Screening check (sample of "not relevant" papers, false-negative rate) | this repo: `src/labelling/screening.py` |
| Overlap kappa (snapshots, bootstrap), judge agreement (population weights, yes-precision) | this repo: `src/labelling/gold.py` (`agreement`, `overlap_from_snapshots`, `judge_agreement`) |
| Review rules | `guidelines/annotation.md` |
| Scorer (levels, matching, vital recall, judged precision) | harness: `packages/evaluation/src/harness_evaluation/matching.py` |
| Entailment rules and closure | harness: `packages/evaluation/src/harness_evaluation/relations.py`, `datasets/ontology/predicate_relations.ttl` |
| Judge and its prompt | harness: `harness_evaluation/judge.py`, `packages/prompts/.../evaluation/triple_match.system.j2` |
| Intervals | harness: `harness_evaluation/stats.py` |
| Benchmark aggregation (per question, complexity × path) | harness: `harness_evaluation/benchmark.py`, `/benchmark` API, `make evaluate` |
| Worked example as a test | harness: `packages/evaluation/tests/test_levels.py` |
| Labelling → harness compatibility | this repo: `tests/test_harness_compatibility.py` |

Commands, in order: `labelling validate` → `evidence-docs` → `triage` → `pool` → `push` →
`screening-sample` → `screening-push` → review → `snapshot` (overlap done, before adjudicating) →
`export` → adjudicate → `export` → `install` → (evaluation runs) → harness `make evaluate ARGS="--judge"`
→ `labelling judge-push` → review the judge sample → `labelling judge-agreement`.

---

## 11. References

Triple and knowledge-graph evaluation

- Mihindukulasooriya, Tiwari, Enguix, Lata. *Text2KGBench: A Benchmark for Ontology-Driven Knowledge Graph Generation from Text.* ISWC 2023. https://arxiv.org/abs/2308.02357
- WebNLG+ Challenge 2020, text-to-RDF evaluation script (strict / exact / partial / type). https://github.com/WebNLG/WebNLG-Text-to-triples
- Zhang, Soh. *Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction.* EMNLP 2024. https://arxiv.org/abs/2404.03868
- Gashteovski et al. *BenchIE: A Framework for Multi-Faceted Fact-Based Open Information Extraction Evaluation.* ACL 2022. https://arxiv.org/abs/2109.06850
- Bhardwaj, Aggarwal, Mausam. *CaRB: A Crowdsourced Benchmark for Open IE.* EMNLP 2019. https://aclanthology.org/D19-1651/
- Euzenat. *Semantic Precision and Recall for Ontology Alignment Evaluation.* IJCAI 2007. https://www.ijcai.org/Proceedings/07/Papers/054.pdf
- Saha et al. *Explanation Graph Generation via Pre-trained Language Models* (G-BERTScore, graph edit distance). ACL 2022. https://arxiv.org/abs/2204.04813
- Mo et al. *KGGen: Extracting Knowledge Graphs from Plain Text with Language Models* (MINE benchmark). 2025. https://arxiv.org/abs/2502.09956
- Tan et al. *Revisiting DocRED – Addressing the False Negative Problem in Relation Extraction.* EMNLP 2022. https://arxiv.org/abs/2205.12696

Pooling, nuggets and judges

- Zobel. *How Reliable are the Results of Large-Scale Information Retrieval Experiments?* SIGIR 1998. https://dl.acm.org/doi/10.1145/290941.291014
- Pradeep et al. *Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework.* 2024. https://arxiv.org/abs/2411.09607
- Pradeep et al. *The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models.* 2025. https://arxiv.org/abs/2504.15068
- Min et al. *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* EMNLP 2023. https://arxiv.org/abs/2305.14251
- Bavaresco et al. *LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks.* ACL 2025. https://arxiv.org/abs/2406.18403
- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (self-enhancement bias). NeurIPS 2023. https://arxiv.org/abs/2306.05685

Question complexity and ambiguity

- Auer et al. *The SciQA Scientific Question Answering Benchmark for Scholarly Knowledge.* Scientific Reports 2023. https://www.nature.com/articles/s41598-023-33607-z
- Banerjee, Awale, Usbeck, Biemann. *DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph.* 2023. https://arxiv.org/abs/2303.13351
- Cao et al. *KQA Pro: A Dataset with Explicit Compositional Programs for Complex Question Answering over Knowledge Base.* ACL 2022. https://aclanthology.org/2022.acl-long.422/
- Trivedi, Balasubramanian, Khot, Sabharwal. *MuSiQue: Multihop Questions via Single-hop Question Composition.* TACL 2022. https://arxiv.org/abs/2108.00573
- Krishna et al. *Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation* (FRAMES). NAACL 2025. https://arxiv.org/abs/2409.12941
- Min, Michael, Hajishirzi, Zettlemoyer. *AmbigQA: Answering Ambiguous Open-domain Questions.* EMNLP 2020. https://arxiv.org/abs/2004.10645
- Stelmakh, Luan, Dhingra, Chang. *ASQA: Factoid Questions Meet Long-Form Answers.* EMNLP 2022. https://arxiv.org/abs/2204.06092
- Jeong, Baek, Cho, Hwang, Park. *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.* NAACL 2024. https://arxiv.org/abs/2403.14403
- Lalor, Wu, Yu. *Building an Evaluation Scale using Item Response Theory.* EMNLP 2016. https://aclanthology.org/D16-1062/
- Rodriguez et al. *Evaluation Examples are not Equally Informative: How should that change NLP Leaderboards?* ACL 2021. https://aclanthology.org/2021.acl-long.346/

Statistics and standards

- Cohen. *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological Measurement, 1960.
- Landis, Koch. *The Measurement of Observer Agreement for Categorical Data.* Biometrics 33(1), 1977. https://doi.org/10.2307/2529310
- Efron, Tibshirani. *An Introduction to the Bootstrap.* Chapman & Hall, 1993.
- W3C. *RDF Schema 1.1.* https://www.w3.org/TR/rdf-schema/ · *OWL 2 Web Ontology Language Profiles* (OWL 2 RL). https://www.w3.org/TR/owl2-profiles/
- OWL-RL reasoner for rdflib (`owlrl`). https://github.com/RDFLib/OWL-RL
