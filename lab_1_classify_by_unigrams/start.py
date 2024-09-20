"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    from  lab_1_classify_by_unigrams import main
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_en:
        en_text = file_en.read()
    en_profile = main.create_language_profile('en', en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_de:
        de_text = file_de.read()
    de_profile = main.create_language_profile('de', de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_unk:
        unknown_text = file_unk.read()
    unk_profile = main.create_language_profile('unk', unknown_text)
    result = None
    assert result, "Detection result is None"

if __name__ == "__main__":
    main()
