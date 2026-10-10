"""Triage and stratified review: a stub model sorts D's triples; pool samples the rest; export estimates misses."""

from __future__ import annotations

import json

from labelling import candidates as cand
from labelling import triage
from labelling.gold import Decision
from labelling.questions import Question

Q = Question(id="q20", category="corpus_wide", question="Which benchmarks does MethodX evaluate on?")


def write_doc(root, triples: int, question: Question = Q, readings: int = 1) -> None:
    folder = root / question.id
    folder.mkdir(parents=True)
    doc = {"question": {"id": question.id, "category": question.category, "question": question.question},
           "interpretations": [
               {"id": f"i{r}", "label": f"reading {r}", "gloss": "",
                "triples": [{"id": f"t{n:05d}", "s": "MethodX", "p": "evaluatedOn", "o": f"Bench{n}",
                             "paper_id": "p97", "sources": [{"source_id": f"p97-{n}", "evidence": f"Bench{n} ..."}]}
                            for n in range(triples)]}
               for r in range(1, readings + 1)]}
    (folder / "doc.json").write_text(json.dumps(doc))


class Model:
    """Sorts by triple number: 0-1 answers, 2 context, the rest irrelevant; fails the first call if asked."""

    def __init__(self, fail_first: bool = False) -> None:
        self.calls, self.fail_first = 0, fail_first

    def complete_json(self, system: str, user: str) -> dict:
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise TimeoutError("slow")
        facts = json.loads(user)["facts"]
        assert "MethodX" in user and "answers" in system
        labels = [{"id": f["id"], "label": ("answers", "answers", "context")[int(f["id"][1:])]
                   if int(f["id"][1:]) < 3 else "irrelevant"} for f in facts]
        return {"labels": labels + [{"id": "t99999", "label": "answers"}, {"id": facts[0]["id"], "label": "maybe"}]}


def test_triage_labels_every_triple_once_and_resumes(tmp_path) -> None:
    write_doc(tmp_path, 25)
    model = Model(fail_first=True)
    counts = triage.triage_doc(tmp_path, "q20", model, batch=10, workers=1, log=lambda _: None)
    assert sum(counts.values()) == 15                  # the failed batch of 10 stays unsorted
    counts = triage.triage_doc(tmp_path, "q20", model, batch=10, workers=1, log=lambda _: None)
    assert counts == {"answers": 2, "context": 1, "irrelevant": 22} and model.calls == 4
    triage.triage_doc(tmp_path, "q20", model, batch=10, workers=1, log=lambda _: None)
    assert model.calls == 4                            # nothing left: no call
    labels = triage.load_labels(tmp_path, "q20")
    assert labels[("i1", "t00002")] == "context" and ("i1", "t99999") not in labels


def test_pool_reviews_the_tiers_and_a_deterministic_sample(corpus, tmp_path) -> None:
    write_doc(tmp_path, 25)
    triage.triage_doc(tmp_path, "q20", Model(), batch=10, workers=1, log=lambda _: None)
    items, _ = cand.from_evidence_docs(corpus, tmp_path, [Q])
    labels = {"q20": triage.load_labels(tmp_path, "q20")}
    chosen, strata = triage.stratify(items, labels, tiers=["answers", "context"], sample=5, seed="s")
    assert [i.stratum for i in chosen].count("triaged") == 3 and [i.stratum for i in chosen].count("sample") == 5
    assert all(i.triage == "irrelevant" for i in chosen if i.stratum == "sample")
    assert strata["q20"] | {"labels": None} == {
        "triaged": True, "tiers": ["answers", "context"], "labels": None, "reviewed_tiers": 3, "unsorted": 0,
        "rest": 22, "sampled": 5, "seed": "s"}
    again, _ = triage.stratify(items, labels, tiers=["answers", "context"], sample=5, seed="s")
    assert [i.candidate_id for i in again] == [i.candidate_id for i in chosen]
    other, _ = triage.stratify(items, labels, tiers=["answers", "context"], sample=5, seed="t")
    assert {i.candidate_id for i in other} != {i.candidate_id for i in chosen}
    # The sample is not grouped at the end, where a reviewer could tell it apart.
    assert [i.stratum for i in chosen] != ["triaged"] * 3 + ["sample"] * 5


def test_unsorted_triples_are_reviewed_and_an_untriaged_question_keeps_everything(corpus, tmp_path) -> None:
    write_doc(tmp_path, 25)
    triage.triage_doc(tmp_path, "q20", Model(fail_first=True), batch=10, workers=1, log=lambda _: None)
    items, _ = cand.from_evidence_docs(corpus, tmp_path, [Q])
    chosen, strata = triage.stratify(items, {"q20": triage.load_labels(tmp_path, "q20")},
                                     tiers=["answers", "context"], sample=0, seed="s")
    assert strata["q20"]["unsorted"] == 10 and len(chosen) == 10      # the failed batch (t0-t9) is reviewed
    chosen, strata = triage.stratify(items, {}, tiers=["answers"], sample=5, seed="s")
    assert len(chosen) == 25 and strata["q20"] == {"triaged": False, "reviewed": 25}
    assert all(i.stratum is None for i in chosen)


def test_readings_keep_their_own_labels(corpus, tmp_path) -> None:
    question = Question(id="q21", category="ambiguous", question="Which MethodX result matters most?")
    write_doc(tmp_path, 4, question, readings=2)
    labels = triage.triage_file(tmp_path, "q21")
    labels.write_text("".join(json.dumps({"reading": r, "triple": "t00000", "label": label}) + "\n"
                              for r, label in [("i1", "answers"), ("i2", "irrelevant")]))
    items, _ = cand.from_evidence_docs(corpus, tmp_path, [question])
    chosen, _ = triage.stratify(items, {"q21": triage.load_labels(tmp_path, "q21")}, tiers=["answers"],
                                sample=0, seed="s")
    tagged = {(i.reading, i.triple_id): i.triage for i in chosen}
    assert tagged[("i1", "t00000")] == "answers" and ("i2", "t00000") not in tagged


def test_export_estimates_what_triage_missed(corpus, tmp_path) -> None:
    write_doc(tmp_path, 25)
    triage.triage_doc(tmp_path, "q20", Model(), batch=10, workers=1, log=lambda _: None)
    items, _ = cand.from_evidence_docs(corpus, tmp_path, [Q])
    chosen, strata = triage.stratify(items, {"q20": triage.load_labels(tmp_path, "q20")},
                                     tiers=["answers", "context"], sample=10, seed="s")
    sample = [i for i in chosen if i.stratum == "sample"]
    decisions = ([Decision(i.candidate_id, True, {}, ["a"], "correct") for i in chosen if i.stratum == "triaged"]
                 + [Decision(i.candidate_id, n < 2, {}, ["a"], "x") for n, i in enumerate(sample)])
    report = triage.miss_report(chosen, decisions, strata)["q20"]
    assert report["sample_reviewed"] == 10 and report["sample_kept"] == 2 and report["tiers_kept"] == 3
    assert report["estimated_missed"] == 2.4                       # 12 unreviewed others x 2/10
    low, high = report["estimated_missed_95"]
    assert 0 < low < 2.4 < high < 12
    assert report["estimated_recall"] == round(3 / (3 + 2 + 2.4), 4)
    recall_low, recall_high = report["estimated_recall_95"]
    share_low, share_high = triage.wilson(2, 10)                   # 12 unreviewed others
    assert recall_low == round(3 / (3 + 2 + 12 * share_high), 4) and recall_high == round(3 / (3 + 2 + 12 * share_low), 4)
    assert report["tiers_unresolved"] == 0 and report["sample_unresolved"] == 0


def test_recall_has_an_interval_when_the_sample_kept_nothing_and_counts_unresolved_records(corpus,
                                                                                           tmp_path) -> None:
    write_doc(tmp_path, 25)
    triage.triage_doc(tmp_path, "q20", Model(), batch=10, workers=1, log=lambda _: None)
    items, _ = cand.from_evidence_docs(corpus, tmp_path, [Q])
    chosen, strata = triage.stratify(items, {"q20": triage.load_labels(tmp_path, "q20")},
                                     tiers=["answers", "context"], sample=10, seed="s")
    tiers = [i for i in chosen if i.stratum == "triaged"]
    sample = [i for i in chosen if i.stratum == "sample"]
    decisions = ([Decision(i.candidate_id, True, {}, ["a"], "correct") for i in tiers[:2]]   # one tier record open
                 + [Decision(i.candidate_id, False, {}, ["a"], "x") for i in sample[:8]])  # two sample records open
    report = triage.miss_report(chosen, decisions, strata)["q20"]
    assert report["estimated_recall"] == 1.0                      # the point estimate alone overstates it
    low, high = report["estimated_recall_95"]
    assert high == 1.0 and low == round(2 / (2 + 14 * triage.wilson(0, 8)[1]), 4) and low < 0.5
    assert report["tiers_unresolved"] == 1 and report["sample_unresolved"] == 2


def test_wilson_interval_bounds() -> None:
    assert triage.wilson(0, 0) == (0.0, 1.0)
    low, high = triage.wilson(0, 50)
    assert low == 0.0 and 0.05 < high < 0.08


def test_pool_command_writes_the_strata_export_reads(corpus, tmp_path, monkeypatch, capsys) -> None:
    from labelling import cli

    write_doc(tmp_path / "docs", 25)
    triage.triage_doc(tmp_path / "docs", "q20", Model(), batch=10, workers=1, log=lambda _: None)
    (tmp_path / "questions.json").write_text(json.dumps([Q.model_dump()]))
    monkeypatch.setenv("CORPUS_ROOT", str(corpus.root))
    out = tmp_path / "out" / "candidates.jsonl"
    assert cli.main(["pool", "--questions", str(tmp_path / "questions.json"), "--docs", str(tmp_path / "docs"),
                     "--out", str(out), "--sample", "4"]) == 0
    assert "q20: 7 of 25 triples to review: 3 in answers+context, 4 sampled from 22 others" in capsys.readouterr().out
    strata = json.loads(out.with_suffix(".strata.json").read_text())
    assert strata["q20"]["rest"] == 22 and len(cand.read_candidates(out)) == 7


def test_draft_answer_key_from_triage_is_marked_and_replaced_by_the_reviewed_one(corpus, tmp_path, monkeypatch,
                                                                                  capsys) -> None:
    import pytest

    pytest.importorskip("harness_extraction", reason="gold rows build claims with the harness")
    from labelling import cli

    question = Question(id="q15", category="corpus_wide", question="What improvement did VPP report on CALVIN?")
    folder = tmp_path / "docs" / "q15"
    folder.mkdir(parents=True)
    triples = [{"id": f"t{n:05d}", "s": "https://dblp.org/rec/conf/icml/Hu25", "p": "reportsMetric", "o": o,
                "paper_id": "p4", "sources": [{"source_id": "p4-results", "evidence": "improves by 41.5%"}]}
               for n, o in enumerate(["41.5% improvement on CALVIN ABC-D", "trained with AdamW", "CALVIN ABC-D"])]
    (folder / "doc.json").write_text(json.dumps({"question": question.model_dump(), "interpretations": [
        {"id": "i1", "label": "As asked", "gloss": "", "triples": triples}]}))
    (folder / "triage.jsonl").write_text("".join(json.dumps({"reading": "i1", "triple": t, "label": label}) + "\n"
                                                 for t, label in [("t00000", "answers"), ("t00001", "irrelevant"),
                                                                  ("t00002", "context")]))
    rows, skipped = triage.draft_rows(corpus, tmp_path / "docs", [question, Question(
        id="q16", category="corpus_wide", question="Not built?")], tiers=["answers", "context"], version="v1",
        ontology_hash="h")
    assert skipped == {"q16": "no evidence document"}
    [row] = rows
    assert row["id"] == "gold-v1-q15" and row["gold_version"] == "v1-draft" and row["reviewer"] == "triage-draft"
    assert sorted(t["o"] for t in row["gold_triples"]) == ["41.5% improvement on CALVIN ABC-D", "CALVIN ABC-D"]
    assert row["status"] == "approved" and triage.is_draft(row)

    harness = tmp_path / "harness"
    monkeypatch.setenv("HARNESS_ROOT", str(harness))
    draft = tmp_path / "draft.jsonl"
    cand.write_jsonl(draft, rows)
    assert cli.main(["install", "--gold", str(draft)]) == 0
    assert "1 DRAFT rows" in capsys.readouterr().out
    reviewed = tmp_path / "reviewed.jsonl"
    cand.write_jsonl(reviewed, [{**row, "gold_version": "v1", "reviewer": "aditya"}])
    assert cli.main(["install", "--gold", str(reviewed)]) == 0
    assert "DRAFT" not in capsys.readouterr().out
    installed = [json.loads(line) for line in (harness / "datasets/gold/candidates.jsonl").read_text().splitlines()]
    assert [(r["id"], r["gold_version"]) for r in installed] == [("gold-v1-q15", "v1")]
