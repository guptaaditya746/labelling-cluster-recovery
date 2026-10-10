"""The question set: corpus-wide and ambiguous questions, and how complex each is.

Complexity is decided before any run, from three facts about the question
(``Complexity``); the level is computed, never typed by hand. The rule and its
reasons are in docs/EVALUATION.md ("Question complexity").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The two kinds of evaluation question (harness datasets/question_set_d.json): corpus-wide questions,
# answered as asked, and ambiguous ones, which the agent interprets (one D section per interpretation).
Category = Literal["corpus_wide", "ambiguous"]

# Operations a correct answer has to perform beyond looking facts up.
Operator = Literal["count", "filter", "intersection", "comparison", "ranking", "negation", "temporal"]
Level = Literal["simple", "compound", "complex"]
LEVELS: tuple[Level, ...] = ("simple", "compound", "complex")


class Complexity(BaseModel):
    """Three facts about what a correct answer needs; the level follows from them.

    - ``papers_read``: papers whose text must be read and combined (0 when DBLP metadata answers
      it; 3 means three or more, or the whole corpus).
    - ``operators``: operations beyond lookup (count, filter, intersection, comparison, ranking,
      negation, temporal).
    - ``readings``: accepted interpretations (2 or more = ambiguous).

    Points: +1 if any paper text is needed; +1 more for two papers, +2 more for three or more;
    +1 per operator; +2 if ambiguous. Level: 0 simple, 1 compound, 2 or more complex.
    """

    model_config = ConfigDict(extra="forbid")
    papers_read: int = Field(ge=0, le=3)
    operators: list[Operator] = Field(default_factory=list)
    readings: int = Field(1, ge=1)

    @property
    def points(self) -> int:
        points = int(self.papers_read >= 1) + len(self.operators)
        points += {2: 1, 3: 2}.get(self.papers_read, 0)
        points += 2 if self.readings >= 2 else 0
        return points

    @property
    def level(self) -> Level:
        return LEVELS[min(self.points, 2)]

    def as_dict(self) -> dict:
        return {"features": self.model_dump(), "points": self.points, "level": self.level}


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    question: str
    category: Category
    type: str | None = None            # what is asked (e.g. corpus_limitations), as in the harness set
    expected: str | None = None        # the expected kind of answer (e.g. evidence_bounded_answer)
    # Facts about what a correct answer needs, decided before any run (two raters); None until rated.
    complexity: Complexity | None = None
    # who asked for it: the professor's set or the student's own
    origin: Literal["professor", "own"] | None = None
    notes: str = ""

    @model_validator(mode="after")
    def consistent(self) -> Question:
        if self.complexity and (self.category == "ambiguous") != (self.complexity.readings >= 2):
            raise ValueError(f"{self.id}: complexity.readings must be 2 or more exactly for ambiguous questions")
        return self

    @property
    def level(self) -> Level | None:
        return self.complexity.level if self.complexity else None


def load_questions(path: Path) -> list[Question]:
    rows = json.loads(Path(path).read_text())
    questions = [Question.model_validate(row) for row in rows]
    ids = [item.id for item in questions]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"duplicate question ids: {duplicates}")
    texts = [" ".join(item.question.casefold().split()) for item in questions]
    if len(set(texts)) != len(texts):
        raise ValueError("duplicate question texts")
    return questions
