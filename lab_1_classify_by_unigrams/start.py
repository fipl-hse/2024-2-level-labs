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
        tokens = tokenize(en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    paths_to_profiles = ['assets/profiles/de.json', 'assets/profiles/en.json',
                         'assets/profiles/es.json', 'assets/profiles/fr.json',
                         'assets/profiles/it.json', 'assets/profiles/ru.json',
                         'assets/profiles/tr.json']
    unk_profile = create_language_profile('unk', unknown_text)
    en_profile = create_language_profile('en', en_text)
    de_profile = create_language_profile('de', de_text)
    if not unk_profile or not en_profile or not de_profile:
        return
    print(tokenize(en_text))
    print(create_language_profile('en', en_text))
    print(detect_language(unk_profile, en_profile, de_profile))
    profiles_collection = collect_profiles(paths_to_profiles)
    if not profiles_collection:
        return
    result = detect_language_advanced(
        unk_profile, profiles_collection)
    if isinstance(result, list):
        print_report(result)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
