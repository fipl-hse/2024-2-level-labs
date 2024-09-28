"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import lab_1_classify_by_unigrams.main as funcs


def main() -> None:
    """
    Launches an implementation
    """
    with open("lab_1_classify_by_unigrams/assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("lab_1_classify_by_unigrams/assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("lab_1_classify_by_unigrams/assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    unknown_profile = funcs.create_language_profile("unk", unknown_text)
    profile_1 = funcs.create_language_profile("en", en_text)
    profile_2 = funcs.create_language_profile("de", de_text)

    if not all(isinstance(prof, dict)
               for prof in (unknown_profile, profile_1, profile_2)):
        result = None
    else:
        result = funcs.detect_language(unknown_profile, profile_1, profile_2)
    assert result, "Detection result is None"
    print(result)


if __name__ == "__main__":
    main()
