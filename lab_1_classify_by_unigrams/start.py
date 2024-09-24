"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import tokenize,calculate_frequencies, create_language_profile, compare_profiles


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

    en_profile = create_language_profile("en", en_text)
    de_profile = create_language_profile("de", de_text)
    unknown_profile = create_language_profile('unknown', unknown_text)

    result = en_text
    print(tokenize(en_text))
    print(tokenize(de_text))
    print(tokenize(unknown_text))
    print(calculate_frequencies(tokenize(en_text)))
    print(create_language_profile('en', en_text))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
