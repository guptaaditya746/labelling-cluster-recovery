# The question set

`questions.json` holds the evaluation questions, the same set as the harness's
`datasets/question_set_d.json` (keep the two equal; the harness's UI and benchmark read its copy):

| Category | Questions | In the evidence document |
|---|---|---|
| `corpus_wide` | nlp32–nlp47 (16) | one section: the question as asked |
| `ambiguous` | nlp30, nlp48–nlp53, nlp62–nlp66 (12) | one section per interpretation the agent proposes |

Before gold is built:

1. **Rate complexity** for every question, by two people independently, before any run (weighted
   kappa), then agree on one value: `papers_read` (3 for corpus-wide), the `operators` a correct answer
   needs, and `readings` (2 or more exactly for ambiguous questions). The level is computed
   (docs/EVALUATION.md, "Question complexity"). Set `origin` (professor / own).
2. Run `labelling validate`: it checks the categories (at least 10 each) and prints the level and
   origin counts and the corpus fingerprint.

## Format

```json
{"id": "nlp32", "category": "corpus_wide", "type": "corpus_limitations",
 "question": "What limitations do the indexed ACL 2025 and EMNLP 2025 papers on retrieval-augmented generation report?",
 "expected": "evidence_bounded_answer",
 "complexity": {"papers_read": 3, "operators": ["filter"]},
 "origin": "own"}
```
