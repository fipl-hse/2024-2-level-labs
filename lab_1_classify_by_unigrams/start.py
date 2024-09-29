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
        en_profile = main.create_language_profile("en", en_text)
        print(en_profile)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_profile = main.create_language_profile("de", de_text)
        print(de_profile)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unk_profile = main.create_language_profile("unk", unknown_text)
        print(unk_profile)
    result = None
    assert result, "Detection result is None"


if __name__ == "__main__":
    run()
