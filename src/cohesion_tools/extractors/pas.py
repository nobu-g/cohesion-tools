from collections import defaultdict
from typing import Dict, List

from rhoknp import BasePhrase
from rhoknp.cohesion import Argument, EndophoraArgument, ExophoraArgument, ExophoraReferentType

from cohesion_tools.extractors.base import BaseExtractor, T


class PasExtractor(BaseExtractor):
    def __init__(
        self,
        cases: List[str],
        exophora_referent_types: List[ExophoraReferentType],
        verbal_predicate: bool = True,
        nominal_predicate: bool = True,
    ) -> None:
        super().__init__(exophora_referent_types)
        self.cases: List[str] = cases
        self.verbal_predicate: bool = verbal_predicate
        self.nominal_predicate: bool = nominal_predicate

    def extract_rels(self, predicate: BasePhrase) -> Dict[str, List[Argument]]:
        all_arguments: Dict[str, List[Argument]] = defaultdict(list)
        candidates: List[BasePhrase] = self.get_candidates(predicate, predicate.document.base_phrases)
        for case in self.cases:
            for argument in predicate.pas.get_arguments(case, relax=False):
                if isinstance(argument, EndophoraArgument):
                    if argument.base_phrase in candidates:
                        all_arguments[case].append(argument)
                elif isinstance(argument, ExophoraArgument):
                    if argument.exophora_referent.type in self.exophora_referent_types:
                        all_arguments[case].append(argument)
                else:
                    raise ValueError(f"argument type {type(argument)} is not supported.")
        return all_arguments

    def is_target(self, base_phrase: BasePhrase) -> bool:
        return self.is_pas_target(base_phrase, verbal=self.verbal_predicate, nominal=self.nominal_predicate)

    @staticmethod
    def is_pas_target(base_phrase: BasePhrase, verbal: bool, nominal: bool) -> bool:
        if verbal and "用言" in base_phrase.features:
            return True
        if nominal and "非用言格解析" in base_phrase.features:
            return True
        return False

    @staticmethod
    def is_candidate(unit: T, predicate: T) -> bool:
        is_anaphora = unit.global_index < predicate.global_index
        is_intra_sentential_cataphora = (
            unit.global_index > predicate.global_index and unit.sentence.sid == predicate.sentence.sid
        )
        return is_anaphora or is_intra_sentential_cataphora
