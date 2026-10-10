"""Validate independent 1–3 ratings and report linear weighted Cohen's kappa."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION_FILE = ROOT / "questions/corpus_10.json"


def read_ratings(path: Path, expected_ids: set[str]) -> dict[str, int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ratings: dict[str, int] = {}
    for row in rows:
        question_id = row.get("question_id", "").strip()
        raw = row.get("difficulty_1_to_3", "").strip()
        if question_id not in expected_ids:
            raise SystemExit(f"{path}: unknown question ID {question_id!r}")
        if question_id in ratings:
            raise SystemExit(f"{path}: duplicate question ID {question_id}")
        if raw not in {"1", "2", "3"}:
            raise SystemExit(f"{path}: {question_id} needs a rating of 1, 2, or 3")
        ratings[question_id] = int(raw)
    missing = expected_ids - ratings.keys()
    if missing:
        raise SystemExit(f"{path}: missing ratings for {sorted(missing)}")
    return ratings


def weighted_kappa(left: dict[str, int], right: dict[str, int]) -> float | None:
    n = len(left)
    if n == 0:
        return None
    categories = (1, 2, 3)
    observed = [[0.0] * 3 for _ in categories]
    left_marginal = [0.0] * 3
    right_marginal = [0.0] * 3
    for question_id, a in left.items():
        b = right[question_id]
        observed[a - 1][b - 1] += 1 / n
        left_marginal[a - 1] += 1 / n
        right_marginal[b - 1] += 1 / n
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(3):
        for j in range(3):
            weight = abs(i - j) / 2  # linear weights: 0 on agreement, 1 on max disagreement
            observed_disagreement += weight * observed[i][j]
            expected_disagreement += weight * left_marginal[i] * right_marginal[j]
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-a", type=Path, default=ROOT / "difficulty/rater_a.csv")
    parser.add_argument("--rater-b", type=Path, default=ROOT / "difficulty/rater_b.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "difficulty/weighted_kappa.json")
    args = parser.parse_args()
    questions = json.loads(QUESTION_FILE.read_text())
    expected_ids = {row["id"] for row in questions}
    a, b = read_ratings(args.rater_a, expected_ids), read_ratings(args.rater_b, expected_ids)
    result = {
        "question_set": "corpus_10",
        "n": len(expected_ids),
        "weighting": "linear",
        "scale": [1, 2, 3],
        "weighted_cohens_kappa": weighted_kappa(a, b),
        "ratings": [
            {"question_id": qid, "rater_a": a[qid], "rater_b": b[qid]}
            for qid in sorted(expected_ids)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"linear weighted Cohen's kappa: {result['weighted_cohens_kappa']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
