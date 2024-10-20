from pathlib import Path

import pytest
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferentType

from cohesion_tools.evaluators.cohesion import CohesionEvaluator
from cohesion_tools.task import Task


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def predicted_documents(data_dir: Path) -> list[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("system/*.knp"))]


@pytest.fixture
def gold_documents(data_dir: Path) -> list[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("gold/*.knp"))]


@pytest.fixture
def scorer() -> CohesionEvaluator:
    return CohesionEvaluator(
        tasks=[Task.PAS_ANALYSIS, Task.BRIDGING_REFERENCE_RESOLUTION, Task.COREFERENCE_RESOLUTION],
        exophora_referent_types=list(map(ExophoraReferentType, ("著者", "読者", "不特定:人", "不特定:物"))),
        pas_cases=["ガ", "ヲ"],
        bridging_rel_types=["ノ"],
    )


@pytest.fixture
def abbreviated_documents(data_dir: Path) -> list[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("knp/*.knp"))]


@pytest.fixture
def restored_documents(data_dir: Path) -> list[Document]:
    return [Document.from_knp(path.read_text()) for path in sorted(data_dir.glob("expected/restored/*.knp"))]
