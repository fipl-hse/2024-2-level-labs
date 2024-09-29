"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from lab_1_classify_by_unigrams.main import (create_language_profile,
                                             collect_profiles, detect_language_advanced, print_report)


def main() -> None:
    """
    Launches an implementation
    """

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        en_prof = create_language_profile('en', en_text)

    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_prof = create_language_profile('de', de_text)

    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_prof = create_language_profile('unknown', unknown_text)

    extra_profiles = ["assets/profiles/de.json",
                      "assets/profiles/en.json",
                      "assets/profiles/es.json",
                      "assets/profiles/fr.json",
                      "assets/profiles/it.json",
                      "assets/profiles/ru.json",
                      "assets/profiles/tr.json"]
    all_profiles = collect_profiles(extra_profiles)
    print_report(detect_language_advanced(unknown_prof, all_profiles))

    return None
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
