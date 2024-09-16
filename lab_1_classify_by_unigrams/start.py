"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from main import tokenize
from main import calculate_frequencies
from main import create_language_profile
from main import compare_profiles
from main import detect_language


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
    print(tokenize(en_text))
    print(calculate_frequencies(tokenize(en_text)))
    print(create_language_profile('en', en_text))
    print(create_language_profile('unk', unknown_text))
    print(compare_profiles(create_language_profile('unk', unknown_text), create_language_profile('en', en_text)))
    print(detect_language(create_language_profile('unk', unknown_text), create_language_profile('en', en_text), create_language_profile('de', de_text)))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
