"""
Language detection starter
"""
#pylint:disable=too-many-locals, unused-argument, unused-variable

from lab_1_classify_by_unigrams.main import create_language_profile, detect_language


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_en:
        en_text = file_en.read()
    en_profile = create_language_profile('english', en_text)
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_de:
        de_text = file_de.read()
    de_profile = create_language_profile('deutch', de_text)
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_unk:
        unknown_text = file_unk.read()
    unk_profile = create_language_profile('unknown', unknown_text)
    if unk_profile and en_profile and de_profile is not None:
        result = detect_language(unk_profile, en_profile, de_profile)
        print(result)
    else:
        result = None
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
