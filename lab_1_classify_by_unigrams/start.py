"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from lab_1_classify_by_unigrams.main import create_language_profile, tokenize


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        text_in_tokens = tokenize(en_text)
        profile_of_en = create_language_profile("en", "en_text")
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        text_in_tokens = tokenize(de_text)
        profile_of_de = create_language_profile("de", "de_text")
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        text_in_tokens = tokenize(unknown_text)
        profile_of_unknown = create_language_profile("un", "un_text")
    result = text_in_tokens
    result_of_six = profile_of_en
    print(result)
    print(result_of_six)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
