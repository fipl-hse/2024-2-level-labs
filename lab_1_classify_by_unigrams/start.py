"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import *

# pylint:disable=too-many-locals, unused-argument, unused-variable


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
    # result = 1  # None
    print(tokenize(en_text))
    print(create_language_profile('en', en_text))
    profile_paths = [
        "assets/profiles/de.json",
        "assets/profiles/en.json",
        "assets/profiles/es.json",
        "assets/profiles/fr.json",
        "assets/profiles/it.json",
        "assets/profiles/ru.json",
        "assets/profiles/tr.json"
    ]
    # assert result, "Detection result is None"


unknown_profile = {
            'name': 'unk',
            'freq': {
                'm': 0.0909, 'e': 0.0909, 'h': 0.1818, 'p': 0.1818,
                'y': 0.0909, 's': 0.0909, 'n': 0.0909, 'a': 0.1818
            }
        }

en_profile = {
    'name': 'en',
    'freq': {
        'p': 0.2, 'y': 0.1, 'e': 0.1, 'h': 0.2,
        'a': 0.2, 'm': 0.1, 'n': 0.1
    }
}

de_profile = {'name': 'de',
              'freq': {
                  'n': 0.0666, 's': 0.0333, 'a': 0.0666, 'm': 0.0666,
                  't': 0.0666, 'i': 0.1333, 'w': 0.0666, 'ß': 0.0333,
                  'ö': 0.0333, 'e': 0.1, 'h': 0.1666, 'c': 0.1666}}

detect_language(unknown_profile, de_profile,en_profile)

if __name__ == "__main__":
    main()
