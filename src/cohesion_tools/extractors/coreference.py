from typing import Union

from rhoknp import BasePhrase
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType

from cohesion_tools.extractors.base import BaseExtractor, T


class CoreferenceExtractor(BaseExtractor):
    def __init__(self, exophora_referent_types: list[ExophoraReferentType]) -> None:
        super().__init__(exophora_referent_types)

    def extract_rels(self, mention: BasePhrase) -> list[Union[BasePhrase, ExophoraReferent]]:
        referents: list[Union[BasePhrase, ExophoraReferent]] = []
        candidates: list[BasePhrase] = self.get_candidates(mention, mention.document.base_phrases)
        for coreferent in mention.get_coreferents(include_nonidentical=False, include_self=False):
            if coreferent in candidates:
                referents.append(coreferent)  # noqa: PERF401
        for exophora_referent in [e.exophora_referent for e in mention.entities if e.exophora_referent is not None]:
            if exophora_referent.type in self.exophora_referent_types:
                referents.append(exophora_referent)  # noqa: PERF401
        return referents

    def is_target(self, mention: BasePhrase) -> bool:
        return self.is_coreference_target(mention)

    @staticmethod
    def is_coreference_target(mention: BasePhrase) -> bool:
        return mention.features.get("体言") is True

    @staticmethod
    def is_candidate(target_mention: T, source_mention: T) -> bool:
        return target_mention.global_index < source_mention.global_index
