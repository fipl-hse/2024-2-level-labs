"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language, detect_language_advanced,
                                             print_report, tokenize)


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        profile_en = create_language_profile("en", en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        profile_de = create_language_profile("de", de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_profile = create_language_profile("unknown", unknown_text)
    result = en_text
    #english: dict = profile_en
    #deutch: dict = profile_de
    #unknown: dict = unknown_profile
    print(tokenize(en_text))
    print(profile_en)

    if unknown_profile and profile_de and profile_en:
        detection = detect_language(unknown_profile, profile_en, profile_de)
        print(detection)

    paths_to_profiles = ['assets/profiles/de.json', 'assets/profiles/en.json',
                          'assets/profiles/es.json', 'assets/profiles/fr.json',
                          'assets/profiles/it.json', 'assets/profiles/ru.json',
                          'assets/profiles/tr.json']

    collection = collect_profiles(paths_to_profiles)
    if collection:
        profiles = detect_language_advanced(unknown_profile, collection)
        if profiles:
            print_report(profiles)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
