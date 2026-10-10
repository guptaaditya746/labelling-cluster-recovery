"""Gold built here must be scored exactly by the harness's own evaluator."""

import pytest

from labelling import gold
from labelling.questions import Question

harness_eval = pytest.importorskip("harness_evaluation", reason="harness packages not installed")


def test_gold_rows_are_valid_and_match_harness_shaped_predictions(corpus) -> None:
    from harness_schema.models import ClaimSubgraph

    question = Question(id="q15", category="corpus_wide", question="What improvement did VPP report?")
    triples = [{"s": "https://dblp.org/rec/conf/icml/Hu25", "p": "reportsMetric",
                "o": "41.5% improvement on CALVIN ABC-D", "support": "human review",
                "evidence": "improves by 41.5%", "source": "p4-results", "weight": "vital", "variants": []}]
    row = gold.gold_row(question, triples, corpus=corpus, version="t", ontology_hash="h", reviewers=["aditya"])
    for claim in row["gold_claims"]:
        ClaimSubgraph.model_validate(claim)
    predicted = [{"s": "https://dblp.org/rec/conf/icml/Hu25", "p": "reportsMetric",
                  "o": "41.5% improvement on CALVIN ABC-D"}]
    score = harness_eval.score_triples(predicted, row["gold_triples"])
    assert (score.precision, score.recall) == (1.0, 1.0)
    claims = harness_eval.score_claims(row["gold_claims"], row["gold_claims"])
    assert claims["claim"]["f1"] == 1.0
    for field in ("gold_triples", "gold_claims", "evidence_refs", "gold_version", "corpus_fingerprint",
                  "ontology_hash", "reviewer", "snapshot"):
        assert row[field], field


def test_an_ambiguous_row_is_scored_against_the_union_of_its_readings(corpus) -> None:
    """AmbigQA set F1: answering one of the two readings recovers half of the gold."""
    from harness_evaluation import evaluate_gold_row
    from harness_schema.models import Metrics, Reading, Run

    question = Question(id="q16", category="ambiguous", question="Which result matters most?")
    triple = {"s": "p4", "p": "reportsMetric", "o": "41.5% improvement", "support": "human review",
              "evidence": "improves by 41.5%", "source": "p4-results", "weight": "okay", "variants": []}
    readings = [{"label": "Largest gain", "measure": "gain", "reading_id": "i1", "gold_triples": [triple]},
                {"label": "Most cited", "measure": "citations", "reading_id": "i2",
                 "gold_triples": [{**triple, "p": "cites", "o": "p5"}]}]
    row = gold.gold_row(question, [], corpus=corpus, version="t", ontology_hash="h", reviewers=["aditya"],
                        accepted_readings=readings)
    run_triples = [{"s": "p4", "p": "reportsMetric", "o": "41.5% improvement", "support": "1 passage",
                    "evidence": "improves", "source": "p4-results"}]
    reading = Reading(label="r", gloss="g", classes=[], axioms=[], sparql="", answer="a", triples=run_triples,
                      metrics=Metrics(), status="answered")
    run = Run(id="r-1", startedAt="2026-01-01T00:00:00Z", durationS=1, question=question.question,
              path="agentic", budget="slow", status="answered", readings=[reading], chosenReading=0)
    result = evaluate_gold_row(run, row)
    soft = result["chosen_levels"]["levels"]["soft"]
    assert (soft["precision"], soft["recall"]) == (1.0, 0.5)
    assert result["levels_gold"] == "readings_union"
    assert result["readings_recall"] == {"Largest gain": 1.0, "Most cited": 0.0}
    assert result["readings_covered"] == 1
