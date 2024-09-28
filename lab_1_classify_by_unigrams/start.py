"""
Language detection starter
"""
from lab_1_classify_by_unigrams.main import tokenize, create_language_profile

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


if __name__ == "__main__":
    main()
