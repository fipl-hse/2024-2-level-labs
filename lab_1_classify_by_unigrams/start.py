"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from lab_1_classify_by_unigrams.main import (tokenize, create_language_profile,
                                             detect_language, collect_profiles,
                                             detect_language_advanced, print_report)


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

    result = en_text
    paths_to_profiles = ['assets/profiles/de.json', 'assets/profiles/en.json',
                         'assets/profiles/es.json', 'assets/profiles/fr.json',
                         'assets/profiles/it.json', 'assets/profiles/ru.json',
                         'assets/profiles/tr.json']
    if (create_language_profile('unk', unknown_text) is not None
            and create_language_profile('en', en_text) is not None
            and create_language_profile('de', de_text) is not None):
        print(tokenize(en_text))
        print(create_language_profile('en', en_text))
        print(detect_language(create_language_profile('unk', unknown_text),
                              create_language_profile('en', en_text),
                              create_language_profile('de', de_text)))
        print_report(detect_language_advanced(create_language_profile(
            'unk', unknown_text), collect_profiles(paths_to_profiles)))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
