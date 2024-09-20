"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import main as funcs

def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_1_classify_by_unigrams/assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("lab_1_classify_by_unigrams/assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("lab_1_classify_by_unigrams/assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    en_lang_profile = funcs.create_language_profile("en", en_text)
    print(en_lang_profile)

    result = None
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
