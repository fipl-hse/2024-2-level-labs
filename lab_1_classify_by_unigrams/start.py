"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import lab_1_classify_by_unigrams.main as mainlab


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        profile_en = mainlab.create_language_profile("en", en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        profile_de = mainlab.create_language_profile("de", de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_profile = mainlab.create_language_profile("unknown", unknown_text)
    if (mainlab.create_language_profile('unk', unknown_text) is None or
    mainlab.create_language_profile('en', en_text) is None or
    mainlab.create_language_profile('de', de_text) is None):
        return None
    print(profile_en)
    detection = mainlab.detect_language(unknown_profile, profile_en, profile_de)
    print(detection)
    paths_to_profiles = [ 'assets/profiles/de.json', 'assets/profiles/en.json',
                          'assets/profiles/es.json', 'assets/profiles/fr.json',
                          'assets/profiles/it.json', 'assets/profiles/ru.json',
                          'assets/profiles/tr.json']
    collection = mainlab.collect_profiles(paths_to_profiles)
    if collection:
        result = mainlab.detect_language_advanced(unknown_profile, collection)
        if not isinstance(result, list):
            return None
        mainlab.print_report(result)
        assert result, "Detection result is None"


if __name__ == "__main__":
    main()
