"""The screening check: a sample of papers D marked not relevant, judged by people (made-up documents)."""

from __future__ import annotations

import json

import pytest

from labelling import argilla_io, cli, screening, triage


def write_doc(root, question_id: str, statuses: dict[str, str], readings: int = 1, category: str = "corpus_wide"):
    folder = root / question_id
    folder.mkdir(parents=True)
    doc = {"question": {"id": question_id, "category": category, "question": "Which methods use FOCUS?"},
           "interpretations": [{"id": f"i{r}", "label": f"reading {r}", "gloss": "a gloss",
                                "coverage": [{"paper_id": pid, "status": status} for pid, status in statuses.items()],
                                "triples": []} for r in range(1, readings + 1)]}
    (folder / "doc.json").write_text(json.dumps(doc))


STATUSES = {"p4": "relevant", "p6": "not_relevant", "p7": "not_relevant", "p9": "not_relevant", "p5": "no_text"}


def test_the_sample_is_seeded_drawn_from_not_relevant_papers_only_and_per_interpretation(corpus, tmp_path) -> None:
    write_doc(tmp_path, "q30", STATUSES)
    write_doc(tmp_path, "q31", STATUSES, readings=2, category="ambiguous")
    records, strata = screening.sample_records(corpus, tmp_path, None, size=2, seed="s")
    assert set(strata) == {"q30", "q31/i1", "q31/i2"}
    assert strata["q30"] | {"papers": None} == {"question_id": "q30", "reading": None, "not_relevant": 3,
                                                "unread": 1, "relevant": 1, "sampled": 2, "papers": None,
                                                "seed": "s"}
    assert all(record["paper_id"] in {"p6", "p7", "p9"} for record in records) and len(records) == 6
    assert records == screening.sample_records(corpus, tmp_path, None, size=2, seed="s")[0]
    first = next(record for record in records if record["question_id"] == "q30")
    assert first["not_relevant_total"] == 3 and first["sample_size"] == 2 and first["reading"] == ""
    assert "`p6-abstract`" in screening.paper_markdown(corpus, "p6")       # every passage of the paper
    ambiguous = next(record for record in records if record["reading"] == "i2")
    assert "Interpretation: reading 2" in ambiguous["question"] and ambiguous["id"].startswith("screening-q31-i2-")
    only, _ = screening.sample_records(corpus, tmp_path, ["q30"], size=10, seed="s")
    assert len(only) == 3                                                   # the whole frame when it is small


def answered(record_id: str, question_id: str, total: int, *verdicts: str, reading: str = "") -> dict:
    return {"id": record_id, "metadata": {"question_id": question_id, "reading": reading, "paper_id": record_id,
                                          "not_relevant_total": total},
            "responses": [{"question": "relevant", "value": value, "user": f"user{n}", "status": "submitted"}
                          for n, value in enumerate(verdicts)]}


def test_the_report_scales_the_wilson_interval_to_the_papers_marked_not_relevant() -> None:
    records = ([answered(f"a{n}", "q30", 200, "no") for n in range(9)] + [answered("a9", "q30", 200, "no", "yes")]
               + [answered(f"b{n}", "q31", 50, "no", reading="i1") for n in range(5)]
               + [{"id": "unanswered", "metadata": {"question_id": "q31", "reading": "i1", "not_relevant_total": 50},
                   "responses": []}])
    result = screening.report({"screening": records})
    q30 = result["sections"]["q30"]
    low, high = triage.wilson(1, 10)
    assert q30["checked"] == 10 and q30["found_relevant"] == 1          # any "yes" counts: the conservative reading
    assert q30["false_negative_rate"] == 0.1 and q30["estimated_missed_papers"] == 20.0
    assert q30["estimated_missed_papers_95"] == [round(200 * low, 1), round(200 * high, 1)]
    assert q30["papers_found_relevant"] == ["a9"]
    q31 = result["sections"]["q31/i1"]
    assert q31["checked"] == 5 and q31["false_negative_rate"] == 0.0 and q31["estimated_missed_papers_95"][1] > 0
    assert result["overall"]["not_relevant"] == 250 and result["overall"]["estimated_missed_papers"] == 20.0
    # N from the strata file when the records carry none.
    plain = [{**record, "metadata": {k: v for k, v in record["metadata"].items() if k != "not_relevant_total"}}
             for record in records[:10]]
    assert screening.report({"screening": plain}, {"q30": {"not_relevant": 40}})["sections"]["q30"][
        "estimated_missed_papers"] == 4.0


class FakeRecords:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def log(self, rows: list[dict]) -> None:
        self.logged += rows


class FakeDataset:
    def __init__(self) -> None:
        self.records = FakeRecords()


def test_screening_commands_write_push_and_report(corpus, tmp_path, monkeypatch, capsys) -> None:
    write_doc(tmp_path / "docs", "q30", STATUSES)
    monkeypatch.setenv("CORPUS_ROOT", str(corpus.root))
    monkeypatch.setenv("HARNESS_DOC_SCREENING_SAMPLE", "2")
    out = tmp_path / "out" / "screening.jsonl"
    assert cli.main(["screening-sample", "--docs", str(tmp_path / "docs"), "--out", str(out)]) == 0
    assert "q30: 2 of 3 papers marked not relevant (1 unread, not in the frame)" in capsys.readouterr().out
    assert json.loads(out.with_suffix(".strata.json").read_text())["q30"]["sampled"] == 2

    datasets: dict[str, FakeDataset] = {}
    monkeypatch.setattr(argilla_io, "client", lambda: object())
    monkeypatch.setattr(argilla_io, "screening_settings", lambda: "settings")   # built by the real SDK elsewhere
    monkeypatch.setattr(argilla_io, "ensure_dataset",
                        lambda api, name, workspace, settings: datasets.setdefault(name, FakeDataset()))
    assert cli.main(["screening-push", "--sample-file", str(out)]) == 0
    logged = datasets["gold-v1-screening"].records.logged
    assert len(logged) == 2 and {"question", "paper", "not_relevant_total", "sample_size"} <= set(logged[0])

    pulled = {"screening": [{"id": row["id"], "metadata": {k: row[k] for k in (
        "question_id", "reading", "paper_id", "not_relevant_total")}, "responses": [
        {"question": "relevant", "value": "no", "user": "aditya", "status": "submitted"}]} for row in logged]}
    monkeypatch.setattr(argilla_io, "pull", lambda **kwargs: pulled)
    report_path = tmp_path / "out" / "screening.report.json"
    assert cli.main(["screening-report", "--sample-file", str(out), "--out", str(report_path)]) == 0
    section = json.loads(report_path.read_text())["sections"]["q30"]
    assert section["checked"] == 2 and section["false_negative_rate"] == 0.0


def test_screening_settings_build_with_the_real_sdk(monkeypatch) -> None:
    from unittest.mock import MagicMock

    import argilla

    monkeypatch.setattr(argilla.Argilla, "_get_default", classmethod(lambda cls: MagicMock()))
    settings = argilla_io.screening_settings()
    assert [q.name for q in settings.questions] == ["relevant", "missing", "note"]
    hidden = {m.name for m in settings.metadata if m.visible_for_annotators is False}
    assert hidden == {"not_relevant_total", "sample_size"}


def test_a_negative_sample_size_is_refused(monkeypatch) -> None:
    from labelling.doc_settings import doc_settings

    monkeypatch.setenv("HARNESS_DOC_SCREENING_SAMPLE", "-1")
    with pytest.raises(ValueError):
        doc_settings()
