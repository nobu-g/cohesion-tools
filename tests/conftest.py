from pathlib import Path

import pytest
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferentType

from cohesion_tools.evaluation import CohesionScorer


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def scorer(data_dir: Path) -> CohesionScorer:
    predicted_documents = [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("system/*.knp"))]
    gold_documents = [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("gold/*.knp"))]

    return CohesionScorer(
        predicted_documents,
        gold_documents,
        exophora_referent_types=[ExophoraReferentType(t) for t in ("著者", "読者", "不特定:人", "不特定:物")],
        pas_cases=["ガ", "ヲ"],
        pas_verbal=True,
        pas_nominal=True,
        bridging=True,
        coreference=True,
    )
