"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from lab_1_classify_by_unigrams.main import (
    create_language_profile,
    detect_language
)

def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        engl_prof = create_language_profile('en', en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        deu_prof = create_language_profile('de', de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unk_prof = create_language_profile('unk', unknown_text)
    result = None
    if isinstance(unk_prof, dict) and isinstance(deu_prof, dict) and isinstance(engl_prof, dict):
        answer = detect_language(unk_prof, engl_prof, deu_prof)
        result = answer
    assert result is not None, "Detection result is None"


if __name__ == "__main__":
    main()
