from typing import List

from rhoknp import Document
from rhoknp.cohesion import ExophoraReferentType

from cohesion_tools.extractors.pas import PasExtractor


def test_to_dict(abbreviated_documents: List[Document], restored_documents: List[Document]) -> None:
    pas_extractor = PasExtractor(
        cases=["ガ", "ヲ", "ニ", "ガ２", "デ", "ト", "カラ", "ヨリ", "マデ", "ヘ", "時間", "外の関係"],
        exophora_referent_types=list(
            map(ExophoraReferentType, ("著者", "読者", "不特定:人", "不特定:物", "不特定:状況"))
        ),
        verbal_predicate=True,
        nominal_predicate=True,
    )
    for abbreviated_document, expected_document in zip(abbreviated_documents, restored_documents):
        actual_document = pas_extractor.restore_pas_annotation(abbreviated_document)
        for actual_base_phrase, expected_base_phrase in zip(
            actual_document.base_phrases, expected_document.base_phrases
        ):
            assert actual_base_phrase.rel_tags == expected_base_phrase.rel_tags
