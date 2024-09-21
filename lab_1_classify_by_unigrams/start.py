from main import (
    tokenize,
    calculate_frequencies,
    create_language_profile,
    calculate_mse,
    compare_profiles,
    detect_language,
    load_profile,
    preprocess_profile,
    collect_profiles,
    detect_language_advanced,
    print_report)

"""
Language detection starter
"""


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
    result = tokenize(en_text)
    print(result)
    print(calculate_frequencies(result))
    print(create_language_profile('en', en_text))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
