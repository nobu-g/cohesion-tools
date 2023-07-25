import io
import json
from pathlib import Path

from cohesion_tools.evaluation import Metric, Scorer


def test_to_dict(data_dir: Path, scorer: Scorer) -> None:
    expected_scores = json.loads(data_dir.joinpath("expected/score/0.json").read_text())
    result = scorer.run().to_dict()
    for task in scorer.pas_cases + ["bridging", "coreference"]:
        task_result = result[task]
        for anal, actual in task_result.items():
            expected: dict = expected_scores[task][anal]
            assert expected["denom_pred"] == actual.tp_fp
            assert expected["denom_gold"] == actual.tp_fn
            assert expected["correct"] == actual.tp


def test_score_result_add(data_dir: Path, scorer: Scorer) -> None:
    expected_scores = json.loads(data_dir.joinpath("expected/score/0.json").read_text())
    score_result1 = scorer.run()
    score_result2 = scorer.run()
    score_result = score_result1 + score_result2
    score_dict = score_result.to_dict()
    for case in scorer.pas_cases:
        case_result = score_dict[case]
        for analysis in Scorer.ARGUMENT_TYPE2ANALYSIS.values():
            expected: dict = expected_scores[case][analysis]
            actual: Metric = case_result[analysis]
            assert actual.tp_fp == expected["denom_pred"] * 2
            assert actual.tp_fn == expected["denom_gold"] * 2
            assert actual.tp == expected["correct"] * 2


def test_export_txt(data_dir: Path, scorer: Scorer) -> None:
    score_result = scorer.run()
    with io.StringIO() as string:
        score_result.export_txt(string)
        string_actual = string.getvalue()
    string_expected = data_dir.joinpath("expected/score/0.txt").read_text()
    assert string_actual == string_expected


def test_export_csv(data_dir: Path, scorer: Scorer) -> None:
    score_result = scorer.run()
    with io.StringIO() as string:
        score_result.export_csv(string)
        string_actual = string.getvalue()
    string_expected = data_dir.joinpath("expected/score/0.csv").read_text()
    assert string_actual == string_expected
