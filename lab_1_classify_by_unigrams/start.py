"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from main import *


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        profile_en = create_language_profile("en", en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        profile_de = create_language_profile("de", de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_profile = create_language_profile("unknown", unknown_text)
    result = detect_language(unknown_profile, profile_en, profile_de)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
