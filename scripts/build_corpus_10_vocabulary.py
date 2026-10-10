"""Refresh the corpus paper-ID aliases in the corpus_10 canonical vocabulary."""

from __future__ import annotations

import json
from pathlib import Path

from labelling.cli import env_path, load_env
from labelling.corpus import Corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_env(ROOT / ".env")
    corpus = Corpus.load(env_path("CORPUS_ROOT", "corpus_data"))
    path = ROOT / "questions/corpus_10_vocabulary.json"
    vocabulary = json.loads(path.read_text())
    aliases = {}
    for paper_id, paper in corpus.papers.items():
        values = [
            paper_id,
            paper.get("title"),
            paper.get("identifier"),
            paper.get("dblp_iri"),
            paper.get("doi"),
            f"https://harness.local/resource/{paper_id}",
        ]
        aliases[paper_id] = list(dict.fromkeys(str(value).strip() for value in values if value))
    vocabulary["paper_aliases"] = aliases
    vocabulary["corpus_fingerprint"] = corpus.fingerprint()
    path.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(aliases)} paper ID mappings to {path}")


if __name__ == "__main__":
    main()
