"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import lab_1_classify_by_unigrams.main as func

def main() -> None:
    """
    Launches an implementation
    """
    list_of_path_to_language_profiles = ["assets/profiles/es.json", "assets/profiles/de.json",
                                         "assets/profiles/en.json", "assets/profiles/fr.json",
                                         "assets/profiles/it.json", "assets/profiles/tr.json",
                                         "assets/profiles/ru.json"]

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
    print(func.tokenize(en_text))
    print(func.create_language_profile("en", en_text))
    en_profile = func.create_language_profile("en", en_text)
    de_profile = func.create_language_profile("de", de_text)
    unknown_profile = func.create_language_profile("unknown", unknown_text)
    if unknown_profile is None or en_profile is None or de_profile is None:
        return None
    print(func.detect_language(unknown_profile, en_profile, de_profile))
    profile_collection = func.collect_profiles(list_of_path_to_language_profiles)
    if profile_collection is None:
        return None
    result = func.detect_language_advanced(unknown_profile, profile_collection)
    if result is None:
        return None
    func.print_report(result)
    assert result, "Detection result is None"
    return None


if __name__ == "__main__":
    main()
