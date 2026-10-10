import json

import pytest

from labelling import candidates as cand
from labelling import gold
from labelling.questions import Question

Q = Question(id="q15", category="corpus_wide", question="What improvement did Video Prediction Policy report on CALVIN?")
A = Question(id="q16", category="ambiguous", question="Which result on CALVIN is the most important?")


def write_doc(root, question, readings):
    """An evidence document as labelling evidence-docs writes it (only the parts review reads)."""
    folder = root / question.id
    folder.mkdir(parents=True)
    doc = {"question": {"id": question.id, "category": question.category, "question": question.question},
           "interpretations": [
               {"id": f"i{n}", "label": label, "gloss": f"{label} gloss",
                "coverage": [{"paper_id": "p4", "status": "relevant"}, {"paper_id": "p5", "status": "not_relevant"}],
                "triples": [{"id": f"t{n:05d}", "s": "https://dblp.org/rec/conf/icml/Hu25", "p": p, "o": o,
                             "paper_id": "p4", "sources": [{"source_id": "p4-results", "evidence": "improves by 41.5%"}]}
                            for p, o in triples]}
               for n, (label, triples) in enumerate(readings, start=1)]}
    (folder / "doc.json").write_text(json.dumps(doc))


def test_candidates_are_the_triples_of_the_evidence_documents(corpus, tmp_path) -> None:
    write_doc(tmp_path, Q, [("As asked", [("reportsMetric", "41.5% improvement on CALVIN ABC-D")])])
    write_doc(tmp_path, A, [("Largest gain", [("reportsMetric", "41.5% improvement on CALVIN ABC-D")]),
                            ("Most cited", [("cites", "p5")])])
    items, missing = cand.from_evidence_docs(corpus, tmp_path, [Q, A, Question(
        id="q17", category="corpus_wide", question="Not built yet?")])
    assert missing == ["q17"]
    plain = next(item for item in items if item.question_id == "q15")
    assert plain.reading is None and plain.triple_id == "t00001" and plain.producers == ["evidence-doc"]
    assert plain.evidence.startswith("Video Prediction Policy improves")        # the passage, not the span
    readings = sorted((item.reading, item.reading_label) for item in items if item.question_id == "q16")
    assert readings == [("i1", "Largest gain"), ("i2", "Most cited")]
    # The same triple under two readings is two records: each reading has its own answer key.
    ids = [item.candidate_id for item in items]
    assert len(ids) == len(set(ids))


def test_an_ambiguous_question_gets_one_accepted_reading_per_interpretation(corpus, tmp_path) -> None:
    write_doc(tmp_path, A, [("Largest gain", [("reportsMetric", "41.5% improvement on CALVIN ABC-D")]),
                            ("Most cited", [("cites", "p5")])])
    items, _ = cand.from_evidence_docs(corpus, tmp_path, [A])
    reviews = {"review": [{"id": item.candidate_id, "responses": review(item.candidate_id, "aditya", "correct",
                                                                          "answers")} for item in items]}
    report = gold.decide(corpus, {item.candidate_id: item for item in items}, reviews)
    added = {("q16", "i2"): [{"s": "p4", "p": "cites", "o": "p6", "source": "p4-results", "evidence": "",
                              "support": "added by aditya", "weight": "okay", "variants": []}]}
    pytest.importorskip("harness_extraction", reason="gold rows build claims with the harness")
    [row] = gold.review_rows([A], items, report, added, corpus=corpus, version="t", ontology_hash="h")
    assert row["ambiguous"] is True and row["gold_triples"] == []
    assert [(r["reading_id"], r["label"], len(r["gold_triples"])) for r in row["accepted_readings"]] == [
        ("i1", "Largest gain", 1), ("i2", "Most cited", 2)]
    assert row["evidence_refs"] == ["p4-results"]


def review(record_id, user, verdict, relevance, correction=None, status="submitted"):
    answers = [("verdict", verdict), ("relevance", relevance)] + ([("correction", correction)] if correction else [])
    return [{"question": q, "value": v, "user": user, "status": status} for q, v in answers]


def candidate(corpus, o="41.5% improvement on CALVIN ABC-D"):
    return cand.Candidate(candidate_id=cand.candidate_id("q15", "p4", "reportsMetric", o, "p4-results"),
                          question_id="q15", question=Q.question, s="https://dblp.org/rec/conf/icml/Hu25",
                          p="reportsMetric", o=o, source="p4-results",
                          evidence=corpus.passages["p4-results"]["text"], paper_id="p4", producers=["evidence-doc"])


def test_decisions_follow_the_rule(corpus) -> None:
    good, fix, bad = candidate(corpus), candidate(corpus, "40% gain"), candidate(corpus, "wrong thing")
    reviews = {"review": [
        {"id": good.candidate_id, "responses": review(good.candidate_id, "aditya", "correct", "answers")},
        {"id": fix.candidate_id, "responses": review(fix.candidate_id, "aditya", "partial", "answers",
                                                      "Video Prediction Policy | reportsMetric | 41.5% improvement on CALVIN ABC-D")},
        {"id": bad.candidate_id, "responses": review(bad.candidate_id, "aditya", "wrong", "answers")},
    ]}
    report = gold.decide(corpus, {c.candidate_id: c for c in (good, fix, bad)}, reviews)
    kept = [d for d in report.decisions if d.keep]
    assert len(kept) == 2 and len(report.decisions) == 3
    fixed = next(d for d in kept if d.reason == "partial, fixed").triple
    assert fixed["s"] == "https://dblp.org/rec/conf/icml/Hu25" and fixed["source"] == "p4-results"


def test_overlap_disagreement_needs_adjudication(corpus) -> None:
    item = candidate(corpus)
    record = {"id": item.candidate_id, "responses": review(item.candidate_id, "aditya", "correct", "answers")
              + review(item.candidate_id, "professor", "wrong", "answers")}
    unresolved = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]})
    assert unresolved.decisions == [] and len(unresolved.disagreements) == 1
    resolved = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]}, adjudicator="professor")
    assert resolved.decisions[0].keep is False and "adjudicated by professor" in resolved.decisions[0].reason


def test_drafts_do_not_count_and_missing_claims_are_parsed(corpus) -> None:
    item = candidate(corpus)
    draft = {"id": item.candidate_id, "responses": review(item.candidate_id, "aditya", "correct", "answers",
                                                          status="draft")}
    assert gold.decide(corpus, {item.candidate_id: item}, {"review": [draft]}).unreviewed == 1
    added, problems = gold.missing_claims(corpus, {"missing": [{
        "id": "missing-q15", "metadata": {"question_id": "q15"},
        "responses": [{"question": "missing", "user": "aditya", "status": "submitted",
                       "value": "p4 | evaluatedOn | CALVIN ABC-D | p4-results\nbroken line"}]}]})
    assert added[("q15", None)][0]["s"] == "https://dblp.org/rec/conf/icml/Hu25"
    assert len(problems) == 1


def test_cohen_kappa() -> None:
    assert gold.cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert gold.cohen_kappa([("a", "b"), ("b", "a")]) == -1.0
    assert gold.cohen_kappa([]) is None
    # Classic example: 20 items, observed agreement 0.7, expected 0.5 -> kappa 0.4
    pairs = [("y", "y")] * 7 + [("n", "n")] * 7 + [("y", "n")] * 3 + [("n", "y")] * 3
    assert gold.cohen_kappa(pairs) == pytest.approx(0.4)


def test_snapshot_mirrors_the_harness_benchmark_defaults(monkeypatch) -> None:
    monkeypatch.delenv("CORPUS_SNAPSHOT_ID", raising=False)
    monkeypatch.setenv("CORPUS_FINGERPRINT", "abc")
    assert gold.snapshot() == {"corpusSnapshotId": "abc", "ontologyVersion": "unknown",
                               "claimSchemaVersion": "v3", "goldVersion": None,
                               "retrievalIndexVersion": "unknown"}


def test_different_fixes_from_two_reviewers_are_a_disagreement(corpus) -> None:
    item = candidate(corpus, "40% gain")
    record = {"id": item.candidate_id, "responses":
              review(item.candidate_id, "aditya", "partial", "answers", "p4 | reportsMetric | 41.5% on CALVIN")
              + review(item.candidate_id, "professor", "partial", "answers", "p4 | reportsMetric | 41.5% gain")}
    report = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]})
    assert report.decisions == [] and len(report.disagreements) == 1


def test_kept_triples_carry_weight_and_variants_and_rejections_keep_their_verdict(corpus) -> None:
    good, bad = candidate(corpus), candidate(corpus, "wrong thing")
    answers = review(good.candidate_id, "aditya", "correct", "answers") + [
        {"question": "weight", "value": "vital", "user": "aditya", "status": "submitted"},
        {"question": "variants", "value": "41.5% gain on CALVIN ABC-D\nVideo Prediction Policy | improves | CALVIN by 41.5%",
         "user": "aditya", "status": "submitted"}]
    reviews = {"review": [{"id": good.candidate_id, "responses": answers},
                          {"id": bad.candidate_id, "responses": review(bad.candidate_id, "aditya", "wrong", "answers")}]}
    report = gold.decide(corpus, {c.candidate_id: c for c in (good, bad)}, reviews)
    kept = next(d for d in report.decisions if d.keep).triple
    assert kept["weight"] == "vital"
    assert kept["variants"] == [
        {"s": "https://dblp.org/rec/conf/icml/Hu25", "p": "reportsMetric", "o": "41.5% gain on CALVIN ABC-D"},
        {"s": "https://dblp.org/rec/conf/icml/Hu25", "p": "improves", "o": "CALVIN by 41.5%"}]
    rejected = next(d for d in report.decisions if not d.keep).triple
    assert (rejected["verdict"], rejected["relevance"]) == ("wrong", "answers")


def test_overlap_agreement_merges_weight_and_variants(corpus) -> None:
    item = candidate(corpus)
    extra = [{"question": "weight", "value": "vital", "user": "aditya", "status": "submitted"},
             {"question": "variants", "value": "41.5% gain", "user": "aditya", "status": "submitted"},
             {"question": "weight", "value": "okay", "user": "professor", "status": "submitted"}]
    record = {"id": item.candidate_id, "responses": review(item.candidate_id, "aditya", "correct", "answers")
              + review(item.candidate_id, "professor", "correct", "supports") + extra}
    report = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]})
    triple = report.decisions[0].triple
    assert triple["weight"] == "okay" and [v["o"] for v in triple["variants"]] == ["41.5% gain"]
    assert gold.agreement({"overlap": [record]})["weight_kappa"] == 0.0   # one item, the two weights differ


def test_missing_claims_accept_a_weight_column(corpus) -> None:
    added, problems = gold.missing_claims(corpus, {"missing": [{
        "id": "missing-q15", "metadata": {"question_id": "q15"},
        "responses": [{"question": "missing", "user": "aditya", "status": "submitted",
                       "value": "p4 | evaluatedOn | CALVIN ABC-D | p4-results | vital\n"
                                "p4 | uses | video | p4-results\n"
                                "p4 | uses | video | p4-results | important"}]}]})
    assert [t["weight"] for t in added[("q15", None)]] == ["vital", "okay"]
    assert len(problems) == 1


def test_judge_agreement_and_balanced_sample() -> None:
    from labelling import cli

    def record(human: str, judge: str) -> dict:
        return {"id": "x", "metadata": {"judge": judge},
                "responses": [{"question": "same_fact", "value": human, "user": "aditya", "status": "submitted"}]}

    reviews = {"judge": [record("yes", "yes")] * 8 + [record("no", "no")] * 8 + [record("no", "yes")] * 2
               + [record("yes", "no")] * 2}
    result = gold.judge_agreement(reviews)
    assert result["items"] == 20 and result["kappa"] == pytest.approx(0.6)
    assert result["use_judge"] is False and result["judge_says_yes_people_say_no"] == 2
    rows = [{"key": f"k{i}", "same_fact": i < 3} for i in range(20)]
    sample = cli.judge_sample(rows, 10, "gold-v1")
    assert len(sample) == 10 and sum(row["same_fact"] for row in sample) == 3
    assert sample == cli.judge_sample(rows, 10, "gold-v1")


def test_weights_in_corrections_and_both_dropped_verdicts(corpus) -> None:
    item = candidate(corpus, "40% gain")
    answers = review(item.candidate_id, "aditya", "partial", "answers",
                     "p4 | reportsMetric | 41.5% improvement on CALVIN ABC-D | vital")
    report = gold.decide(corpus, {item.candidate_id: item}, {"review": [{"id": item.candidate_id, "responses": answers}]})
    fixed = report.decisions[0].triple
    assert fixed["weight"] == "vital" and fixed["source"] == "p4-results"
    record = {"id": item.candidate_id, "responses": review(item.candidate_id, "aditya", "wrong", "answers")
              + review(item.candidate_id, "professor", "correct", "irrelevant")}
    dropped = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]}).decisions[0]
    assert dropped.keep is False and dropped.triple["verdict"] == "mixed"


def test_the_ontology_hash_covers_the_dblp_seed_and_the_cskg_subset(tmp_path, monkeypatch) -> None:
    from labelling import cli

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "dblp_schema_2024-06-14.ttl").write_text("# dblp")
    monkeypatch.setenv("HARNESS_ROOT", str(tmp_path))
    assert cli.ontology_hash() == "unknown"                     # the CS-KG subset is missing
    (seed / "cskg_ontology_2.0_subset.ttl").write_text("# cskg")
    first = cli.ontology_hash()
    (seed / "cskg_ontology_2.0_subset.ttl").write_text("# cskg, changed")
    assert len(first) == 64 and cli.ontology_hash() != first


def test_papers_get_the_identifier_runs_use_and_ids_keep_d_spelling(corpus, tmp_path) -> None:
    """D's extraction names a paper by its corpus key; runs name it by its DBLP IRI. Gold must match runs."""
    folder = tmp_path / Q.id
    folder.mkdir()
    triples = [{"id": "t00001", "s": "p4", "p": "cites", "o": "p6", "paper_id": "p4",
                "sources": [{"source_id": "p4-results", "evidence": "x"}]},
               {"id": "t00002", "s": "MethodX", "p": "extends", "o": "p4", "paper_id": "p4",
                "sources": [{"source_id": "p4-results", "evidence": "x"}]}]
    (folder / "doc.json").write_text(json.dumps({"question": {"id": Q.id, "category": Q.category,
                                                              "question": Q.question},
                                                 "interpretations": [{"id": "i1", "triples": triples}]}))
    first, second = cand.from_evidence_docs(corpus, tmp_path, [Q])[0]
    assert (first.s, first.o) == ("https://dblp.org/rec/conf/icml/Hu25", "p6")      # p6 has no DBLP IRI
    assert (second.s, second.o) == ("MethodX", "https://dblp.org/rec/conf/icml/Hu25")
    assert first.candidate_id == cand.candidate_id(Q.id, "p4", "cites", "p6", "p4-results")


# ─── Judge agreement: stratified sample, population weights, one response per record ───

def judge_record(record_id: str, judge: str, **answers: str) -> dict:
    return {"id": record_id, "metadata": {"judge": judge},
            "responses": [{"question": "same_fact", "value": value, "user": user, "status": "submitted"}
                          for user, value in answers.items()]}


def balanced_judge_sample() -> list[dict]:
    """30 judge-yes records (people say yes to 24) and 30 judge-no records (people say no to 27)."""
    return ([judge_record(f"y{i}", "yes", aditya="yes" if i < 24 else "no") for i in range(30)]
            + [judge_record(f"n{i}", "no", aditya="no" if i < 27 else "yes") for i in range(30)])


def test_judge_kappa_is_reweighted_to_the_population_and_decides_on_the_weighted_value() -> None:
    records = balanced_judge_sample()
    manifest = {"population": {"yes": 150, "no": 850}, "sample": {"yes": 30, "no": 30},
                "record_ids": {"yes": [f"y{i}" for i in range(30)], "no": [f"n{i}" for i in range(30)]}}
    result = gold.judge_agreement({"judge": records}, manifest)
    assert result["raw_kappa"] == pytest.approx(0.7)
    # Weighted table (x5 for judge-yes, x850/30 for judge-no): 120 / 30 / 85 / 765 of 1,000.
    observed, expected = 0.885, 0.205 * 0.15 + 0.795 * 0.85
    assert result["weighted_kappa"] == pytest.approx((observed - expected) / (1 - expected))
    assert result["kappa"] == result["weighted_kappa"] and result["kappa_basis"].startswith("population")
    assert result["use_judge"] is False                       # the raw 0.70 would have passed the bar
    assert result["weighted_agreement"] == pytest.approx(0.885)
    precision = result["precision_judge_yes"]
    assert precision["rate"] == 0.8 and precision["n"] == 30 and precision["ci95"][0] < 0.8 < precision["ci95"][1]
    assert result["person_yes_given_judge_no"]["rate"] == 0.1
    assert result["warnings"] == []


def test_judge_agreement_without_a_manifest_falls_back_to_raw_kappa_with_a_warning() -> None:
    result = gold.judge_agreement({"judge": balanced_judge_sample()})
    assert result["kappa"] == pytest.approx(0.7) and result["weighted_kappa"] is None
    assert result["use_judge"] is True and "no judge-push manifest" in result["warnings"][0]


def test_judge_agreement_uses_one_response_per_record_and_reports_each_person() -> None:
    records = [judge_record("r1", "yes", bea="yes", cem="no"), judge_record("r2", "no", bea="no", cem="no"),
               judge_record("r3", "yes", cem="yes"), judge_record("old", "no", bea="yes")]
    manifest = {"population": {"yes": 2, "no": 1}, "record_ids": {"yes": ["r1", "r3"], "no": ["r2"]}}
    result = gold.judge_agreement({"judge": records}, manifest)
    assert result["items"] == 3 and result["responses"] == 5          # r1 counts once (bea, first by name)
    assert result["judge_says_yes_people_say_no"] == 0
    assert result["per_person"]["bea"]["items"] == 2 and result["per_person"]["cem"]["items"] == 3
    assert result["per_person"]["bea"]["raw_kappa"] == 1.0
    assert any("not in the latest push's manifest" in warning for warning in result["warnings"])  # "old"


def test_judge_push_records_the_population_and_judge_agreement_reads_it(tmp_path, monkeypatch, capsys) -> None:
    from labelling import argilla_io, cli

    cache = tmp_path / "judge_cache.jsonl"
    rows = [{"key": f"key{i:04d}" + "x" * 30, "same_fact": i < 10, "prompt": "v2",
             "gold": {"subject": "MethodX", "relation": "uses", "object": "ToolQ"},
             "run": {"subject": "MethodX", "relation": "uses", "object": f"Tool{i}"}} for i in range(100)]
    cache.write_text("".join(json.dumps(row) + "\n" for row in rows))
    pushed: list[dict] = []
    monkeypatch.setattr(argilla_io, "push_judge", lambda chosen, **_: pushed.extend(chosen) or len(chosen))
    manifest_path = tmp_path / "judge.manifest.json"
    assert cli.main(["judge-push", "--cache", str(cache), "--sample", "20", "--manifest", str(manifest_path)]) == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["population"] == {"yes": 10, "no": 90} and manifest["sample"] == {"yes": 10, "no": 10}
    assert sorted(manifest["record_ids"]["yes"] + manifest["record_ids"]["no"]) == sorted(
        argilla_io.judge_record_id(row) for row in pushed)
    answered = [judge_record(argilla_io.judge_record_id(row), "yes" if row["same_fact"] else "no",
                             aditya="yes" if row["same_fact"] else "no") for row in pushed]
    monkeypatch.setattr(argilla_io, "pull", lambda **_: {"judge": answered})
    assert cli.main(["judge-agreement", "--manifest", str(manifest_path)]) == 0
    printed = capsys.readouterr().out
    assert '"kappa_basis": "population-weighted (N_h/n_h)"' in printed and '"use_judge": true' in printed


# ─── Overlap agreement: the export's rule, dropped records, intervals, snapshots ───

def overlap_record(record_id: str, question_id: str, **answers: tuple) -> dict:
    responses = []
    for user, answer in answers.items():
        verdict, relevance, *correction = answer
        responses += review(record_id, user, verdict, relevance, correction[0] if correction else None)
    return {"id": record_id, "metadata": {"question_id": question_id}, "responses": responses}


def test_agreement_counts_a_partial_without_a_fix_as_dropped_like_the_export(corpus) -> None:
    item = candidate(corpus, "40% gain")
    record = overlap_record(item.candidate_id, "q15", aditya=("partial", "answers"),
                            professor=("partial", "answers", "p4 | reportsMetric | 41.5% on CALVIN"))
    result = gold.agreement({"overlap": [record]})
    assert result["raw_keep_agreement"] == 0.0                   # drop (no fix) vs keep (fixed)
    report = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]})
    assert len(report.disagreements) == 1                       # the export sees the same disagreement
    assert gold.would_keep({"verdict": "partial", "relevance": "supports", "correction": "a | b | c"})
    assert not gold.would_keep({"verdict": "partial", "relevance": "supports", "correction": "a | b"})


def test_agreement_reports_dropped_records_and_a_question_bootstrap_interval() -> None:
    records = []
    for q in range(4):
        for n in range(6):
            same = n != q % 3                                   # one disagreement in most questions
            records.append(overlap_record(f"r{q}-{n}", f"q{q}", aditya=("correct", "answers"),
                                          professor=("correct" if same else "wrong", "answers")))
    records += [overlap_record("solo", "q0", aditya=("wrong", "answers")),
                overlap_record("pair", "q1", aditya=("correct", "answers"), carol=("correct", "answers"))]
    result = gold.agreement({"overlap": records})
    assert result["reviewers"] == ["aditya", "professor"] and result["items"] == 24 and result["questions"] == 4
    assert result["records_dropped_one_reviewer"] == 1 and result["records_dropped_other_reviewers"] == 1
    low, high = result["keep_kappa_95"]
    assert -1 <= low <= result["keep_kappa"] <= high <= 1
    assert result["bootstrap"]["questions"] == 4 and result == gold.agreement({"overlap": records})  # fixed seed
    one_question = gold.agreement({"overlap": [r for r in records if r["metadata"]["question_id"] == "q0"]})
    assert one_question["keep_kappa_95"] is None


def test_kappa_comes_from_the_earliest_snapshot_not_from_responses_edited_in_adjudication() -> None:
    before = [overlap_record("r1", "q1", aditya=("correct", "answers"), professor=("wrong", "answers")),
              overlap_record("r2", "q1", aditya=("wrong", "answers"), professor=("wrong", "answers"))]
    after = [overlap_record("r1", "q1", aditya=("correct", "answers"), professor=("correct", "answers")),
             before[1], overlap_record("r3", "q1", aditya=("correct", "answers"), professor=("correct", "answers"))]
    merged, source = gold.overlap_from_snapshots([{"taken_at": "t0", "path": "s0.json", "overlap": before}], after)
    assert source == {"snapshots": ["s0.json"], "records_from_snapshots": 2, "records_from_live": 1}
    assert gold.agreement({"overlap": merged})["raw_keep_agreement"] == pytest.approx(2 / 3)
    assert gold.agreement({"overlap": after})["raw_keep_agreement"] == 1.0


def test_an_adjudicator_who_is_one_of_the_raters_is_flagged(corpus) -> None:
    item = candidate(corpus)
    record = {"id": item.candidate_id, "responses": review(item.candidate_id, "aditya", "correct", "answers")
              + review(item.candidate_id, "professor", "wrong", "answers")}
    own = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [record]}, adjudicator="professor")
    assert own.adjudicated == 1 and own.adjudicated_by_a_rater == 1
    assert "one of the two raters" in own.decisions[0].reason
    third = {**record, "responses": record["responses"] + review(item.candidate_id, "carol", "correct", "answers")}
    other = gold.decide(corpus, {item.candidate_id: item}, {"overlap": [third]}, adjudicator="carol")
    assert other.adjudicated == 1 and other.adjudicated_by_a_rater == 0 and other.decisions[0].keep is True


def test_export_snapshots_the_overlap_and_warns_about_a_rater_adjudicator(corpus, tmp_path, monkeypatch,
                                                                           capsys) -> None:
    pytest.importorskip("harness_extraction", reason="gold rows build claims with the harness")
    from labelling import argilla_io, cli

    monkeypatch.setenv("CORPUS_ROOT", str(corpus.root))
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps([Q.model_dump()]))
    good, other = candidate(corpus), candidate(corpus, "trained with AdamW")
    candidates = tmp_path / "out" / "candidates.jsonl"
    cand.write_jsonl(candidates, [good, other])
    independent = {"overlap": [
        overlap_record(good.candidate_id, "q15", aditya=("correct", "answers"), professor=("correct", "answers")),
        overlap_record(other.candidate_id, "q15", aditya=("correct", "answers"), professor=("wrong", "answers"))]}
    resolved = {"overlap": [independent["overlap"][0], overlap_record(
        other.candidate_id, "q15", aditya=("correct", "answers"), professor=("correct", "answers"))]}
    pulls = iter([independent, resolved])
    monkeypatch.setattr(argilla_io, "pull", lambda **_: next(pulls))
    out = tmp_path / "out" / "gold-v1.jsonl"
    args = ["export", "--questions", str(questions), "--candidates", str(candidates), "--out", str(out),
            "--adjudicator", "professor"]
    assert cli.main(args) == 0
    first = json.loads(out.with_suffix(".report.json").read_text())
    assert first["adjudication"]["adjudicator_is_overlap_rater"] is True
    assert first["adjudication"]["resolved_by_a_rater"] == 1
    assert "WARNING: adjudication" in capsys.readouterr().err
    assert len(list((tmp_path / "out" / "snapshots").glob("gold-v1-overlap-*.json"))) == 1
    assert cli.main(args) == 0                                   # after the edit that "resolved" it
    second = json.loads(out.with_suffix(".report.json").read_text())
    assert second["agreement"]["raw_keep_agreement"] == 0.5      # the independent round, from the snapshot
    assert second["agreement_live"]["raw_keep_agreement"] == 1.0
    assert second["agreement_source"]["records_from_snapshots"] == 2


# ─── Gold triples: one per fact, every passage kept ───

def test_the_same_fact_from_two_passages_is_one_gold_triple_with_both_sources(corpus) -> None:
    base = {"s": "https://dblp.org/rec/conf/icml/Hu25", "p": "reportsMetric", "o": "41.5% improvement on CALVIN",
            "support": "human review", "evidence": "improves by 41.5%"}
    triples = [{**base, "source": "p4-results", "weight": "okay", "variants": [{"s": "a", "p": "b", "o": "c"}]},
               {**base, "o": "41.5%  Improvement on CALVIN", "source": "p4-abstract", "weight": "vital",
                "variants": [{"s": "a", "p": "b", "o": "d"}]},
               {**base, "o": "something else", "source": "p4-results", "weight": "okay", "variants": []}]
    [merged, different] = gold.dedupe(triples)
    assert merged["source"] == "p4-results" and merged["sources"] == ["p4-results", "p4-abstract"]
    assert merged["weight"] == "vital" and [v["o"] for v in merged["variants"]] == ["c", "d"]
    assert different["sources"] == ["p4-results"]
    pytest.importorskip("harness_extraction", reason="gold rows build claims with the harness")
    row = gold.gold_row(Q, triples, corpus=corpus, version="t", ontology_hash="h", reviewers=["aditya"])
    assert len(row["gold_triples"]) == 2 and row["evidence_refs"] == ["p4-abstract", "p4-results"]


def test_facts_added_on_screening_records_become_gold(corpus) -> None:
    added, problems = gold.missing_claims(corpus, {"screening": [{
        "id": "screening-q15-p6", "metadata": {"question_id": "q15", "reading": "", "paper_id": "p6"},
        "responses": [{"question": "relevant", "user": "aditya", "status": "submitted", "value": "yes"},
                      {"question": "missing", "user": "aditya", "status": "submitted",
                       "value": "p6 | models | object interactions | p6-abstract | vital"}]}]})
    assert problems == [] and added[("q15", None)][0]["weight"] == "vital"
