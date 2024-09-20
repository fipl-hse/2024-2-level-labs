from main import tokenize
from main import calculate_frequencies
from main import create_language_profile
from main import calculate_mse
from main import compare_profiles
from main import detect_language
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
    result = 0
    assert result, "Detection result is None"
    print(tokenize(en_text))


if __name__ == "__main__":
    main()

