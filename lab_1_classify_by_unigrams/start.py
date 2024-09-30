"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
import main

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
    result = print(main.create_language_profile('enflish', en_text))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
