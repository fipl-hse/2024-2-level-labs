"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import main as func


def main() -> None:
    """
    Launches an i mplementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    result = func.create_language_profile("en", en_text)
    assert result, "Detection result is None"
    print(result)



if __name__ == "__main__":
    main()

