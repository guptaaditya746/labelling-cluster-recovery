from pathlib import Path

import pytest

from labelling.corpus import Corpus

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def corpus() -> Corpus:
    return Corpus.load(FIXTURES / "corpus")
