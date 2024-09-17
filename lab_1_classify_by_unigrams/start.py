from lab_1_classify_by_unigrams.main import (tokenize,
                                             create_language_profile,
                                             compare_profiles,
                                             detect_language,
                                             calculate_mse,
                                             load_profile)
"""
Language detection starter
"""


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
    result = 1  # None
    print(tokenize(en_text))
    print(create_language_profile('en', en_text))
    assert result, "Detection result is None"
    profile_paths = [
        "assets/profiles/de.json",
        "assets/profiles/en.json",
        "assets/profiles/es.json",
        "assets/profiles/fr.json",
        "assets/profiles/it.json",
        "assets/profiles/ru.json",
        "assets/profiles/tr.json"
    ]
    # print(load_profile(profile_paths[0]))


unknown_profile = {
    'name': 'unk',
    'freq': {
        'e': 0.2222, 'h': 0.0555, 'c': 0.0555, 't': 0.0555,
        'r': 0.0555, 'g': 0.0555, 'b': 0.0555, 'w': 0.0555,
        'l': 0.0555, 'ß': 0.0555, 'i': 0.1111, 'ü': 0.0555, 'n': 0.1111
    }
}

en_profile = {
    'name': 'en',
    'freq': {
        'p': 0.2, 'y': 0.1, 'e': 0.1, 'h': 0.2,
        'a': 0.2, 'm': 0.1, 'n': 0.1
    }
}
de_profile = {
    'name': 'de',
    'freq': {
        't': 0.0833, 'h': 0.1666, 'n': 0.0833, 'w': 0.0833,
        'ß': 0.0833, 'e': 0.0833, 'c': 0.1666, 'i': 0.25
    }
}

print(detect_language(unknown_profile, en_profile, de_profile))

if __name__ == "__main__":
    main()
