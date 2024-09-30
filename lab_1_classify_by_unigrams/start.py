"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from main import tokenize
def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        text_in_tokens = tokenize(en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        text_in_tokens = tokenize(de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        text_in_tokens = tokenize(unknown_text)
    result = text_in_tokens
    print(result)
    assert result, "Detection result is None"



if __name__ == "__main__":
    main()
