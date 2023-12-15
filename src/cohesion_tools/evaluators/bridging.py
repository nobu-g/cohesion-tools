import copy
from typing import Callable, ClassVar, Collection, Dict, List, Optional

import pandas as pd
from rhoknp import Document
from rhoknp.cohesion import Argument, ArgumentType, EndophoraArgument, ExophoraArgument, ExophoraReferentType, Predicate

from cohesion_tools.evaluators.utils import F1Metric


class BridgingReferenceResolutionEvaluator:
    """橋渡し参照解析の評価を行うクラス

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`rhoknp.Document`

    Args:
        exophora_referent_types: 評価の対象とする外界照応の照応先 (rhoknp.cohesion.ExophoraReferentTypeType を参照)
    """

    ARGUMENT_TYPE_TO_ANALYSIS_TYPE: ClassVar[Dict[ArgumentType, str]] = {
        ArgumentType.CASE_EXPLICIT: "dep",
        ArgumentType.CASE_HIDDEN: "dep",
        ArgumentType.OMISSION: "zero_endophora",
        ArgumentType.EXOPHORA: "exophora",
    }

    def __init__(
        self,
        exophora_referent_types: Collection[ExophoraReferentType],
        rel_types: Collection[str],
    ) -> None:
        self.exophora_referent_types: List[ExophoraReferentType] = list(exophora_referent_types)
        self.rel_types: List[str] = list(rel_types)
        self.anaphor_filter: Callable[[Predicate], bool] = lambda _: True
        self.comp_result: Dict[tuple, str] = {}

    def run(self, predicted_document: Document, gold_document: Document) -> pd.Series:
        """Compute bridging reference resolution scores"""
        predicted_anaphors = [base_phrase.pas.predicate for base_phrase in predicted_document.base_phrases]
        gold_anaphors = [base_phrase.pas.predicate for base_phrase in gold_document.base_phrases]
        metrics: Dict[str, F1Metric] = {anal: F1Metric() for anal in ("dep", "zero_endophora", "exophora")}

        assert len(predicted_anaphors) == len(gold_anaphors)
        for predicted_anaphor, gold_anaphor in zip(predicted_anaphors, gold_anaphors):
            if self.anaphor_filter(predicted_anaphor) is True:
                predicted_antecedents: List[Argument] = self._filter_referents(
                    predicted_anaphor.pas.get_arguments("ノ", relax=False),
                    predicted_anaphor,
                )
            else:
                predicted_antecedents = []
            # Assuming one antecedent for one anaphor
            assert len(predicted_antecedents) in (0, 1)

            if self.anaphor_filter(gold_anaphor) is True:
                gold_antecedents: List[Argument] = self._filter_referents(
                    gold_anaphor.pas.get_arguments("ノ", relax=False),
                    gold_anaphor,
                )
                relaxed_gold_antecedents: List[Argument] = gold_anaphor.pas.get_arguments(
                    "ノ",
                    relax=True,
                    include_nonidentical=True,
                )
                relaxed_gold_antecedents += gold_anaphor.pas.get_arguments("ノ？", relax=True, include_nonidentical=True)
                relaxed_gold_antecedents = self._filter_referents(relaxed_gold_antecedents, gold_anaphor)
            else:
                gold_antecedents = relaxed_gold_antecedents = []

            key = (predicted_anaphor.base_phrase.global_index, "ノ")

            # Compute precision
            if len(predicted_antecedents) > 0:
                predicted_antecedent = predicted_antecedents[0]
                if predicted_antecedent in relaxed_gold_antecedents:
                    relaxed_gold_antecedent = relaxed_gold_antecedents[
                        relaxed_gold_antecedents.index(predicted_antecedent)
                    ]
                    analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[relaxed_gold_antecedent.type]
                    self.comp_result[key] = analysis
                    metrics[analysis].tp += 1
                else:
                    analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[predicted_antecedent.type]
                    self.comp_result[key] = "wrong"
                metrics[analysis].tp_fp += 1

            # Compute recall
            if gold_antecedents or (self.comp_result.get(key, None) in self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE.values()):
                recalled_antecedent: Optional[Argument] = None
                for relaxed_gold_antecedent in relaxed_gold_antecedents:
                    if relaxed_gold_antecedent in predicted_antecedents:
                        recalled_antecedent = relaxed_gold_antecedent  # 予測されている先行詞を優先して正解の先行詞に採用
                        break
                if recalled_antecedent is not None:
                    analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[recalled_antecedent.type]
                    if analysis == "overt":
                        analysis = "dep"
                    assert self.comp_result[key] == analysis
                else:
                    analysis = self.ARGUMENT_TYPE_TO_ANALYSIS_TYPE[gold_antecedents[0].type]
                    if analysis == "overt":
                        analysis = "dep"
                    if len(predicted_antecedents) > 0:
                        assert self.comp_result[key] == "wrong"
                    else:
                        self.comp_result[key] = "wrong"
                metrics[analysis].tp_fn += 1
        return pd.Series(metrics)

    def _filter_referents(self, referents: List[Argument], anaphor: Predicate) -> List[Argument]:
        filtered = []
        for orig_referent in referents:
            referent = copy.copy(orig_referent)
            if referent.case.endswith("≒"):
                referent.case = referent.case[:-1]
            if referent.case == "ノ？":
                referent.case = "ノ"
            if isinstance(referent, ExophoraArgument):
                referent.exophora_referent.index = None  # 「不特定:人１」なども「不特定:人」として扱う
                if referent.exophora_referent.type not in self.exophora_referent_types:
                    continue
            else:
                assert isinstance(referent, EndophoraArgument)
                # Filter out self-anaphora
                if referent.base_phrase == anaphor.base_phrase:
                    continue
                # Filter out cataphora
                if (
                    referent.base_phrase.global_index > anaphor.base_phrase.global_index
                    and referent.base_phrase.sentence.sid != anaphor.base_phrase.sentence.sid
                ):
                    continue
            filtered.append(referent)
        return filtered
