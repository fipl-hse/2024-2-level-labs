"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable


from lab_1_classify_by_unigrams.main import create_language_profile, detect_language


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        profile_to_compare_en = create_language_profile('en', en_text)

    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        profile_to_compare_de = create_language_profile('de', de_text)

    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_profile = create_language_profile('unknown', unknown_text)

    if (isinstance(unknown_profile, dict)
            and isinstance(profile_to_compare_en, dict)
            and isinstance(profile_to_compare_de, dict)):

        answer = detect_language(unknown_profile, profile_to_compare_en, profile_to_compare_de)

        result = answer
        print(result)
        assert result, "Detection result is None"


if __name__ == "__main__":
    main()
