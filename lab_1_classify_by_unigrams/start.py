"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import (tokenize, calculate_frequencies, create_language_profile)
# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        #tokenize(en_text)
        print(create_language_profile('english',en_text))
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        tokenize(de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        tokenize(unknown_text)
    result = tokenize
    assert result, "Detection result is None"



if __name__ == "__main__":
    main()
