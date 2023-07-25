from typing import List

from rhoknp import BasePhrase, Document
from rhoknp.cohesion import ExophoraReferentType


class BaseExtractor:
    def __init__(self, exophora_referent_types: List[ExophoraReferentType]) -> None:
        self.exophora_referent_types = exophora_referent_types

    def extract_rels(self, document: Document):
        raise NotImplementedError

    def is_target(self, base_phrase: BasePhrase) -> bool:
        raise NotImplementedError

    @staticmethod
    def is_candidate(bp: BasePhrase, anaphor: BasePhrase) -> bool:
        raise NotImplementedError

    def get_candidates(self, base_phrase: BasePhrase) -> List[BasePhrase]:
        return [bp for bp in base_phrase.document.base_phrases if self.is_candidate(bp, base_phrase) is True]
