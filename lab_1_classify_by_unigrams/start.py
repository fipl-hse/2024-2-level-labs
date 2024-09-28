"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from lab_1_classify_by_unigrams.main import create_language_profile, detect_language, tokenize


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
    result = 1
    unknown_language_profile = create_language_profile('unknown',unknown_text)
    en_language_profile = create_language_profile('en',en_text)
    de_language_profile = create_language_profile('de',de_text)
    print(tokenize(en_text))
    print(create_language_profile('en',en_text))
    print(detect_language(unknown_language_profile,en_language_profile,de_language_profile))

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
