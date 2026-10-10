"""Read-only view of the harness corpus, using the harness's identifier rules.

Gold triples must use exactly the identifiers the harness emits, otherwise the
evaluator's set comparison never matches: papers are their verified DBLP IRI
when one exists, else their local key (``p4``); predicates are the harness
names (``title``, ``authoredBy``, ``year``, ``venue``, ``cites``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

CORPUS_FILES = ("papers.jsonl", "passages.jsonl", "facts.jsonl", "references.jsonl")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


@dataclass
class Corpus:
    root: Path
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)
    passages: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, root: Path) -> Corpus:
        root = Path(root).expanduser()
        missing = [name for name in CORPUS_FILES if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(f"corpus at {root} is missing {', '.join(missing)}")
        return cls(
            root=root,
            papers={str(row["paper_id"]): row for row in read_jsonl(root / "papers.jsonl")},
            passages={str(row["passage_id"]): row for row in read_jsonl(root / "passages.jsonl")},
            facts=read_jsonl(root / "facts.jsonl"),
            references=read_jsonl(root / "references.jsonl"),
        )

    # ─── identity ────────────────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """SHA-256 over the four corpus files: gold is tied to this exact snapshot."""
        digest = hashlib.sha256()
        for name in CORPUS_FILES:
            digest.update(name.encode())
            digest.update((self.root / name).read_bytes())
        return digest.hexdigest()

    def paper_iri(self, paper_id: str) -> str:
        """The identifier the harness emits for a paper (DBLP IRI, else local key)."""
        paper = self.papers.get(str(paper_id))
        iri = paper.get("dblp_iri") if paper else None
        return str(iri) if iri else str(paper_id)

    def title(self, paper_id: str) -> str:
        for row in self.facts:
            if row.get("predicate") == "title" and str(row.get("subject")) == str(paper_id):
                return str(row["object"])
        paper = self.papers.get(str(paper_id)) or {}
        return str(paper.get("title") or paper_id)

    def resolve_paper(self, mention: str) -> str | None:
        """Map a name or title mention to one paper key, or None when unclear."""
        wanted = norm(mention)
        if not wanted:
            return None
        if wanted in self.papers or mention in self.papers:
            return mention
        titles = {pid: norm(self.title(pid)) for pid in self.papers}
        exact = [pid for pid, title in titles.items() if title == wanted]
        if len(exact) == 1:
            return exact[0]
        heads = [pid for pid in self.papers if norm(self.title(pid).split(":", 1)[0]) == wanted]
        if len(heads) == 1:
            return heads[0]
        scored = sorted(((SequenceMatcher(None, wanted, title).ratio(), pid) for pid, title in titles.items()),
                        reverse=True)
        if scored and scored[0][0] >= 0.8 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.1):
            return scored[0][1]
        return None
