from pathlib import Path

import pytest
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent

from cohesion_tools.evaluation import Scorer


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def scorer(data_dir: Path) -> Scorer:
    predicted_documents = [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("system/*.knp"))]
    gold_documents = [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("gold/*.knp"))]

    return Scorer(
        predicted_documents,
        gold_documents,
        exophora_referents=[ExophoraReferent(e) for e in ("著者", "読者", "不特定:人", "不特定:物")],
        pas_cases=["ガ", "ヲ"],
        pas_target="all",
        bridging=True,
        coreference=True,
    )
