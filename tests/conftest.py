from pathlib import Path
from typing import List

import pytest
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferentType

from cohesion_tools.evaluation import CohesionEvaluator


@pytest.fixture()
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture()
def predicted_documents(data_dir: Path) -> List[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("system/*.knp"))]


@pytest.fixture()
def gold_documents(data_dir: Path) -> List[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("gold/*.knp"))]


@pytest.fixture()
def scorer() -> CohesionEvaluator:
    return CohesionEvaluator(
        exophora_referent_types=list(map(ExophoraReferentType, ("著者", "読者", "不特定:人", "不特定:物"))),
        pas_cases=["ガ", "ヲ"],
        bridging=True,
        coreference=True,
    )


@pytest.fixture()
def abbreviated_documents(data_dir: Path) -> List[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("knp/*.knp"))]


@pytest.fixture()
def restored_documents(data_dir: Path) -> List[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("expected/restored/*.knp"))]
