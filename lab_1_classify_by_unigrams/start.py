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
        # en_token = tokenize(en_text)
        en_profile = create_language_profile("en", en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_profile = create_language_profile("de", de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unk_profile = create_language_profile("unk", unknown_text)

    result = unk_profile, en_profile, de_profile
    print(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
