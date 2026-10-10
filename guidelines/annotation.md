# Annotation guideline (gold v1)

You judge the **claims (triples) of one question's evidence document (D)**, built by screening every
paper for the question and extracting the relevant papers' passages for it. You see the triples a model
sorted as answers or context plus a random sample of the others, shuffled; you are not told which is
which. For an ambiguous question each record also names the **interpretation** it
belongs to: judge relevance against that interpretation. Judge each claim **only against its evidence
passage**; do not use outside knowledge. Why each field exists, and how it is scored, is explained in `docs/EVALUATION.md`.

## 1. Is the claim correct? (`verdict`)

- **Correct**: the passage states it (paraphrase is fine), and the subject, relation and object are
  all right.
- **Partly correct**: the core is right but one part is wrong or too vague (for example a missing
  benchmark, wrong number, wrong subject). Write the fixed claim in `correction` as
  `subject | relation | object`.
- **Wrong**: the passage does not support it, or it contradicts the passage.

Numbers must match the passage exactly (41.5%, not "about 40%"). A result without its benchmark
or metric is at most **partly correct**. Wrong claims are kept on record: the harness uses them to
measure how often a system states something false (judged precision).

## 2. Is it relevant? (`relevance`)

- **Answers the question**: part of the answer itself.
- **Needed context**: not the answer, but required to interpret it (metric, benchmark, setting,
  comparator, the paper that reports it).
- **Not relevant**: true perhaps, but not needed for this question.

Only claims that are correct or fixed **and** relevant become gold: they are the question's answer
key.

## 3. For a kept claim: weight and other wordings

- **Weight** (`weight`): **Vital** if an answer without this fact misses the point of the question;
  **Okay** if it is worth having but a good answer could leave it out. Most questions have 1–4 vital
  facts. Leave it empty for claims you reject; an empty weight on a kept claim counts as okay.
  Vital recall ("did the answer get the essential facts?") is computed from these marks.
- **Other acceptable wordings** (`variants`): one per line, other ways a correct answer could state
  the same fact. Write just the object ("RAG vulnerability to manipulated retrieved knowledge"), or a
  whole claim (`subject | relation | object`) when the relation changes too. Two or three typical
  wordings are enough; the scorer's "soft" level uses them before any model judge.

You no longer need to keep the candidate's exact wording when fixing a claim: write the clearest
correct claim and put the other good wordings in `variants`.

## 4. Subjects

When a claim is the paper's own statement ("this paper reports ..."), the subject is the **paper**:
write its title or key (`2025-acl-long-230`); the export maps it to the paper's identifier.

## 5. Missing claims (the `-missing` dataset)

After reviewing a question's claims, open its record in the `-missing` dataset (one per interpretation
of an ambiguous question). It lists the papers D found relevant. Add any correct, relevant claim the
document does not contain, one per line:

```
subject | relation | object | passage_id | vital
```

The last column is optional (`vital` or `okay`, default okay). Use the passage ids listed in the
record, or any passage of the corpus. Mark "Reviewed" even if nothing is missing. This step is what keeps
the gold from being only what the builder found.

**Papers D marked not relevant (the `-screening` dataset).** A random sample of the papers the screening
marked not relevant, a few per question (and per interpretation), is loaded here
(`labelling screening-sample`, `labelling screening-push`). Each record shows the question and every
passage of the paper. Answer: **is this paper relevant to the question** (or the interpretation)? Yes
if it states a fact that answers the question or is needed to read the answer, judged as in section 2.
If yes, add the facts D is missing in `missing`, one per line, in the format above (passage ids of this
paper). Do not answer from the title alone; skim the passages. Answer No when the paper only mentions
the topic. The sample is random on purpose: check every record, also the ones that look obviously
irrelevant, or the estimate of what screening missed (`labelling screening-report`) is biased. Facts you
add here become gold like the missing claims above.

## 6. The overlap subset

About 20% of the questions are in the `-overlap` dataset. Both annotators review them
**independently** (do not discuss before submitting). When both have finished, run `labelling snapshot`
**before** discussing anything: kappa is computed from the earliest snapshot in which both answers of a
record are present (every `labelling export` saves one too), so editing a response to settle a
disagreement does not raise it. Agreement is reported as Cohen's kappa for verdict, relevance, keep/drop
(the export's own rule: a partly-correct claim without a usable fix counts as dropped) and weight, with a
bootstrap interval over questions. Disagreements are then resolved by the **adjudicator**: a third
person who submits their own answer on the disputed records, or the outcome of a recorded discussion
entered under a separate user. If the adjudicator is one of the two raters, that rater's answer simply
wins; the export warns and records it (`adjudication` in the report). A claim both keep is vital only if
both marked it vital; their wordings are pooled.

## 7. Checking the judge (the `-judge` dataset)

After the harness has scored runs with its soft-match judge, a sample of the judge's decisions is
loaded here (`labelling judge-push`). For each pair, answer: does the candidate state the same fact as
the reference? Yes only if a reader of the candidate learns the reference fact; vaguer, more general
or different claims are No. You do not see the judge's answer. The sample holds about as many pairs the
judge accepted as pairs it rejected, which is not their share among all its decisions; `labelling
judge-agreement` reweights them to the real shares and reports that kappa (and each annotator's own);
below 0.61 the soft level is reported without the judge.

## 8. When unsure

Choose **Partly correct** or **Wrong** rather than Correct, **Okay** rather than Vital, and write a
short note.
