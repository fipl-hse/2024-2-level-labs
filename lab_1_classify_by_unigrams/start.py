"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (create_language_profile, detect_language)
# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    result = detect_language
    en_text_1 = create_language_profile('english', en_text)
    de_text_1 = create_language_profile('deutsch', de_text)
    unknown_text_1 = create_language_profile('unknown', unknown_text)
    if not all(isinstance(text, dict) for text in [unknown_text_1, en_text_1, de_text_1]):
        result = None
    print(detect_language(unknown_text_1, en_text_1, de_text_1))
    assert result, "Detection result is None"



if __name__ == "__main__":
    main()
