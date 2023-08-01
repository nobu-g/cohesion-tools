import io
import json
from pathlib import Path
from typing import List

from rhoknp import Document

from cohesion_tools.evaluation import CohesionScorer, Metrics


def test_to_dict(
    data_dir: Path, predicted_documents: List[Document], gold_documents: List[Document], scorer: CohesionScorer
) -> None:
    expected_scores = json.loads(data_dir.joinpath("expected/score/0.json").read_text())
    score = scorer.run(predicted_documents, gold_documents).to_dict()
    for task in list(scorer.pas_cases) + ["bridging", "coreference"]:
        task_result = score[task]
        for anal, actual in task_result.items():
            expected: dict = expected_scores[task][anal]
            assert expected["denom_precision"] == actual.tp_fp
            assert expected["denom_recall"] == actual.tp_fn
            assert expected["tp"] == actual.tp


def test_score_addition(
    data_dir: Path, predicted_documents: List[Document], gold_documents: List[Document], scorer: CohesionScorer
) -> None:
    expected_scores = json.loads(data_dir.joinpath("expected/score/0.json").read_text())
    score1 = scorer.run(predicted_documents, gold_documents)
    score2 = scorer.run(predicted_documents, gold_documents)
    score = score1 + score2
    score_dict = score.to_dict()
    for case in scorer.pas_cases:
        case_result = score_dict[case]
        for analysis in CohesionScorer.ARGUMENT_TYPE2ANALYSIS.values():
            expected: dict = expected_scores[case][analysis]
            actual: Metrics = case_result[analysis]
            assert actual.tp_fp == expected["denom_precision"] * 2
            assert actual.tp_fn == expected["denom_recall"] * 2
            assert actual.tp == expected["tp"] * 2


def test_identical_document(data_dir: Path, gold_documents: List[Document], scorer: CohesionScorer) -> None:
    score = scorer.run(gold_documents, gold_documents)
    score_dict = score.to_dict()
    for value1 in score_dict.values():
        for value2 in value1.values():
            assert value2.tp_fp == value2.tp_fn == value2.tp


def test_export_txt(
    data_dir: Path, predicted_documents: List[Document], gold_documents: List[Document], scorer: CohesionScorer
) -> None:
    score = scorer.run(predicted_documents, gold_documents)
    with io.StringIO() as string:
        score.export_txt(string)
        string_actual = string.getvalue()
    string_expected = data_dir.joinpath("expected/score/0.txt").read_text()
    assert string_actual == string_expected


def test_export_csv(
    data_dir: Path, predicted_documents: List[Document], gold_documents: List[Document], scorer: CohesionScorer
) -> None:
    score = scorer.run(predicted_documents, gold_documents)
    with io.StringIO() as string:
        score.export_csv(string)
        string_actual = string.getvalue()
    string_expected = data_dir.joinpath("expected/score/0.csv").read_text()
    assert string_actual == string_expected
