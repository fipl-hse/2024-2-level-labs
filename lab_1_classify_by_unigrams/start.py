"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from main import *

list_of_path_to_language_profiles = ["assets/profiles/es.json", "assets/profiles/de.json",
                                     "assets/profiles/en.json", "assets/profiles/fr.json",
                                     "assets/profiles/it.json", "assets/profiles/tr.json", "assets/profiles/ru.json"]


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
    print(tokenize(en_text))
    print(create_language_profile("en", en_text))
    en_profile = create_language_profile("en", en_text)
    de_profile = create_language_profile("de", de_text)
    unknown_profile = create_language_profile("unknown", unknown_text)
    print(detect_language(unknown_profile, en_profile, de_profile))
    profile_collection = collect_profiles(list_of_path_to_language_profiles)
    result = detect_language_advanced(unknown_profile, profile_collection)
    print_report(result)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
