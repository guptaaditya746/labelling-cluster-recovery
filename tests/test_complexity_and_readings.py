"""Question complexity (computed from three facts) and the corpus-wide / ambiguous question set."""

import json
from pathlib import Path

import pytest

from labelling import cli
from labelling.questions import Complexity, Question, load_questions


@pytest.mark.parametrize(("features", "points", "level"), [
    ({"papers_read": 0}, 0, "simple"),
    ({"papers_read": 0, "operators": ["count"]}, 1, "compound"),
    ({"papers_read": 1}, 1, "compound"),
    ({"papers_read": 2, "operators": ["comparison"]}, 3, "complex"),
    ({"papers_read": 3, "operators": ["ranking"], "readings": 2}, 6, "complex"),
])
def test_complexity_points_and_level(features, points, level) -> None:
    complexity = Complexity(**features)
    assert (complexity.points, complexity.level) == (points, level)
    assert complexity.as_dict() == {"features": complexity.model_dump(), "points": points, "level": level}


def test_only_ambiguous_questions_have_several_readings() -> None:
    wide = Question(id="q", question="Which limitations recur?", category="corpus_wide",
                    complexity=Complexity(papers_read=3, operators=["filter"]))
    assert wide.level == "complex"
    Question(id="q", question="Which approach is best?", category="ambiguous",
             complexity=Complexity(papers_read=3, operators=["ranking"], readings=3))
    with pytest.raises(ValueError, match="ambiguous"):
        Question(id="q", question="Which approach is best?", category="ambiguous",
                 complexity=Complexity(papers_read=3))
    with pytest.raises(ValueError, match="corpus_wide"):
        Question(id="q", question="Q?", category="synthesis")


def test_the_repository_question_set_is_the_evidence_document_set() -> None:
    questions = load_questions(Path(__file__).parents[1] / "questions" / "questions.json")
    categories = {item.category for item in questions}
    assert categories == {"corpus_wide", "ambiguous"}
    assert all(sum(item.category == c for item in questions) >= 10 for c in categories)


def test_validate_reports_categories_and_unrated_complexity(corpus, tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("CORPUS_ROOT", str(corpus.root))
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([
        {"id": "a", "category": "corpus_wide", "question": "What limitations recur?",
         "complexity": {"papers_read": 3}, "origin": "own"},
        {"id": "b", "category": "ambiguous", "question": "Which method is best?"},
    ]))
    code = cli.main(["validate", "--questions", str(path)])
    out = capsys.readouterr().out
    assert code == 1                                            # fewer than 10 per category
    assert "complexity: simple=0, compound=0, complex=1, unset=1" in out
    assert "1 questions have no complexity yet" in out
    assert "category ambiguous has 1 questions" in out
