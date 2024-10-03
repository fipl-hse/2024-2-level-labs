"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from lab_1_classify_by_unigrams.main import (calculate_frequencies, create_language_profile,
                                             detect_language, tokenize)


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
    print(tokenize(en_text))
    print(calculate_frequencies(tokenize(en_text)))
    print(create_language_profile('en', en_text))
    en_profile = create_language_profile('en', en_text)
    du_profile = create_language_profile('de', de_text)
    un_profile = create_language_profile('un', unknown_text)
    if en_profile is None or du_profile is None or un_profile is None:
        return
    result = detect_language(un_profile, du_profile, en_profile)
    if result is None:
        print("failed")
    else:
        print(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
