"""
Language detection starter
"""
from main import tokenize, calculate_frequencies


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
    result = tokenize(en_text)
    print(result)
    print(calculate_frequencies(result))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
