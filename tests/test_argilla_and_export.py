import json
from types import SimpleNamespace

import pytest

from labelling import argilla_io, cli
from labelling import candidates as cand
from labelling.questions import load_questions


def test_argilla_settings_build_with_the_real_sdk(monkeypatch) -> None:
    # The SDK resolves a default client (which connects); stand in for the server.
    from unittest.mock import MagicMock

    import argilla

    monkeypatch.setattr(argilla.Argilla, "_get_default", classmethod(lambda cls: MagicMock()))
    review = argilla_io.review_settings(2)
    assert [q.name for q in review.questions] == ["verdict", "relevance", "weight", "variants", "correction", "note"]
    assert next(q for q in review.questions if q.name == "weight").required is False
    assert review.distribution.min_submitted == 2
    assert {m.name for m in review.metadata} == {"question_id", "reading", "paper", "producers"}
    assert [q.name for q in argilla_io.missing_settings().questions] == ["missing", "done"]
    judge = argilla_io.judge_settings()
    assert [q.name for q in judge.questions] == ["same_fact"]
    assert next(m for m in judge.metadata if m.name == "judge").visible_for_annotators is False


def test_overlap_sample_is_deterministic_and_about_a_fifth(tmp_path) -> None:
    rows = [{"id": f"s{i}", "category": "corpus_wide", "question": f"Q{i}?"} for i in range(18)]
    path = tmp_path / "q.json"
    path.write_text(json.dumps(rows))
    questions = load_questions(path)
    first = cli.overlap_ids(questions, "auto", "gold-v1")
    assert len(first) == 4 and first == cli.overlap_ids(questions, "auto", "gold-v1")


def test_offline_export_writes_the_reviewed_answer_keys(corpus, tmp_path, monkeypatch) -> None:
    pytest.importorskip("harness_extraction", reason="harness packages not installed")
    monkeypatch.setenv("CORPUS_ROOT", str(corpus.root))
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([
        {"id": "q02", "category": "corpus_wide", "question": "Who wrote VPP?"},
        {"id": "q15", "category": "corpus_wide", "question": "What improvement did VPP report?"},
    ]))
    questions = load_questions(questions_path)
    item = cand.Candidate(candidate_id="cand-1", question_id="q15", question="What improvement did VPP report?",
                          s="https://dblp.org/rec/conf/icml/Hu25", p="reportsMetric",
                          o="41.5% improvement on CALVIN ABC-D", source="p4-results",
                          evidence=corpus.passages["p4-results"]["text"], paper_id="p4", producers=["evidence-doc"])
    reviews = {"review": [{"id": "cand-1", "responses": [
        {"question": "verdict", "value": "correct", "user": "aditya", "status": "submitted"},
        {"question": "relevance", "value": "answers", "user": "aditya", "status": "submitted"}]}]}
    out = tmp_path / "gold.jsonl"
    args = SimpleNamespace(version="v1", adjudicator=None, out=str(out))
    assert cli.export_from_reviews(args, corpus, questions, [item], reviews) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["gold-v1-q15"]           # q02 has no reviewed triple
    assert rows[0]["gold_triples"][0]["o"] == "41.5% improvement on CALVIN ABC-D"
    assert rows[0]["gold_claims"][0]["kind"] == "ResultClaim"
    report = json.loads(out.with_suffix(".report.json").read_text())
    assert report["kept"] == 1 and report["unresolved_disagreements"] == 0


def test_install_refuses_a_missing_or_empty_gold_file(tmp_path, monkeypatch) -> None:
    from labelling.cli import main

    harness = tmp_path / "harness"
    target = harness / "datasets" / "gold" / "candidates.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text('{"id": "gold-keep", "status": "approved"}\n')
    monkeypatch.setenv("HARNESS_ROOT", str(harness))
    assert main(["install", "--gold", str(tmp_path / "missing.jsonl")]) == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert main(["install", "--gold", str(empty)]) == 2
    assert target.read_text() == '{"id": "gold-keep", "status": "approved"}\n'
    assert not list(target.parent.glob("*.bak"))


def test_an_unreachable_argilla_explains_the_tunnel(monkeypatch) -> None:
    import pytest

    from labelling import argilla_io

    monkeypatch.setenv("ARGILLA_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ARGILLA_API_KEY", "test-key")
    with pytest.raises(SystemExit) as stopped:
        argilla_io.client()
    assert "omen-tunnels" in str(stopped.value)
