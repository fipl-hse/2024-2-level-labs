"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import main


def run() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        print(main.tokenize(en_text))
        tokens = main.tokenize(en_text)
        print(main.calculate_frequencies(tokens))
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        #print(main.tokenize(de_text))
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        #print(main.tokenize(unknown_text))
    result = None
    assert result, "Detection result is None"


if __name__ == "__main__":
    run()
