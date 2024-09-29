"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from lab_1_classify_by_unigrams.main import (collect_profiles, create_language_profile,
                                             detect_language_advanced, detect_language,
                                             load_profile, preprocess_profile,
                                             print_report, tokenize)


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        tokens = tokenize(en_text)
        en_profile = create_language_profile("eng", en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_profile = create_language_profile("de", de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unk_profile = create_language_profile("unknown", unknown_text)

    print(create_language_profile("eng", en_text))
    if en_profile is not None and de_profile is not None and unk_profile is not None:
        print(detect_language(en_profile, de_profile, unk_profile))

    processed_profile = preprocess_profile(load_profile('assets/profiles/de.json'))
    print(processed_profile)

    collected_profiles = collect_profiles(['assets/profiles/de.json', 'assets/profiles/en.json',
                                           'assets/profiles/es.json', 'assets/profiles/fr.json',
                                           'assets/profiles/it.json', 'assets/profiles/ru.json',
                                           'assets/profiles/tr.json'])

    if detect_language_advanced is not None and print_report is not None:
        print_report(detect_language_advanced(unk_profile, collected_profiles))

    result = 1

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
