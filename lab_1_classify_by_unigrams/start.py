"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language,detect_language_advanced,
                                             print_report, tokenize)


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
    paths_to_profiles = ['assets/profiles/de.json', 'assets/profiles/en.json',
                         'assets/profiles/es.json', 'assets/profiles/fr.json',
                         'assets/profiles/it.json', 'assets/profiles/ru.json',
                         'assets/profiles/tr.json']
    if unknown_language_profile and en_language_profile and de_language_profile:
        print(tokenize(en_text))
        print(create_language_profile('en', en_text))
        print(detect_language(unknown_language_profile, en_language_profile, de_language_profile))
        collection_of_profiles = collect_profiles(paths_to_profiles)
        if collection_of_profiles:
            detection = detect_language_advanced(unknown_language_profile, collection_of_profiles)
            if isinstance(detection, list):
                print_report(detection)


    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
