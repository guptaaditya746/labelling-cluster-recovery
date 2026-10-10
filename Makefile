.PHONY: setup harness evidence-docs triage draft validate pool push export install judge-push judge-agreement test

# --inexact keeps the harness packages `make harness` installed; a plain `uv sync` removes them
# (ModuleNotFoundError: harness_agent) and `make harness` must be run again.
setup:
	uv sync --group dev --inexact

# The harness packages from a checkout pinned at a tag (git -C $HARNESS_ROOT checkout <tag>): build_claims and
# the evaluator for gold rows, and readers, concept readings and extraction for the evidence documents.
HARNESS_PACKAGES = schema config prompts semantic extraction evaluation corpus graphstore ledger observability baseline agent
harness:
	uv pip install $(foreach p,$(HARNESS_PACKAGES),-e $${HARNESS_ROOT:?set HARNESS_ROOT}/packages/$(p))

# One exhaustive evidence document (D) per question; resumable. See docs/EVIDENCE_DOCS.md.
# make evidence-docs ARGS="--only nlp33"   ·   ARGS="--status"   ·   ARGS="--only nlp33 --out <earlier folder>"
evidence-docs:
	uv run labelling evidence-docs $(ARGS)

# A model sorts each D triple (answers / context / irrelevant); pool then takes the first two and a sample.
# make triage ARGS="--only nlp33"   ·   make pool ARGS="--sample 50"   ·   make pool ARGS="--all"
triage:
	uv run labelling triage $(ARGS)
# Until the review is done: a DRAFT answer key from triage alone, then `make install ARGS="--gold out/gold-v1.draft.jsonl"`.
draft:
	uv run labelling draft $(ARGS)

validate:
	uv run labelling validate
pool:
	uv run labelling pool $(ARGS)
push:
	uv run labelling push
export:
	uv run labelling export $(if $(ADJUDICATOR),--adjudicator $(ADJUDICATOR),)
install:
	uv run labelling install $(ARGS)
judge-push:  ## sample the harness judge's decisions into Argilla (after `make evaluate ARGS=--judge` there)
	uv run labelling judge-push
judge-agreement:  ## kappa people vs judge; decides whether soft scores may use the judge
	uv run labelling judge-agreement
test:
	uv run ruff check . && uv run pytest -q
