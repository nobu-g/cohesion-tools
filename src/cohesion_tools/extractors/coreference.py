from rhoknp import BasePhrase
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType

from cohesion_tools.extractors.base import BaseExtractor, T


class CoreferenceExtractor(BaseExtractor):
    def __init__(self, exophora_referent_types: list[ExophoraReferentType]) -> None:
        super().__init__(exophora_referent_types)

    def extract_rels(self, base_phrase: BasePhrase) -> list[BasePhrase | ExophoraReferent]:
        referents: list[BasePhrase | ExophoraReferent] = []
        candidates: list[BasePhrase] = self.get_candidates(base_phrase, base_phrase.document.base_phrases)
        for coreferent in base_phrase.get_coreferents(include_nonidentical=False, include_self=False):
            if coreferent in candidates:
                referents.append(coreferent)  # noqa: PERF401
        for exophora_referent in [e.exophora_referent for e in base_phrase.entities if e.exophora_referent is not None]:
            if exophora_referent.type in self.exophora_referent_types:
                referents.append(exophora_referent)  # noqa: PERF401
        return referents

    def is_target(self, base_phrase: BasePhrase) -> bool:
        return self.is_coreference_target(base_phrase)

    @staticmethod
    def is_coreference_target(mention: BasePhrase) -> bool:
        return mention.features.get("体言") is True

    @staticmethod
    def is_candidate(possible_candidate: T, anaphor: T) -> bool:
        return possible_candidate.global_index < anaphor.global_index
