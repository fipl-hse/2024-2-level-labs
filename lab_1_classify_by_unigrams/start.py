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
        tokens = tokenize(en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    print(create_language_profile("eng", en_text))
    print(detect_language(create_language_profile("unknown", unknown_text),
                          create_language_profile("en", en_text),
                          create_language_profile("de", de_text)))
    result = tokens

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
