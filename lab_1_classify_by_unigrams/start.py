"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
def main() -> None:
    """
    Launches an implementation
    """
    from lab_1_classify_by_unigrams.main import create_language_profile, detect_language

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        en_prof = create_language_profile('en', en_text)

    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        de_prof = create_language_profile('de', de_text)

    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        unknown_prof = create_language_profile('unknown', unknown_text)

    extra_profiles = ["lab_1_classify_by_unigrams\\assets\\profiles\\de.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\en.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\es.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\fr.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\it.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\ru.json"
                      "lab_1_classify_by_unigrams\\assets\\profiles\\tr.json"]

    result = detect_language(unknown_prof, en_prof, de_prof)
    return result
    #assert result, "Detection result is None"


if __name__ == "__main__":
    print(main())

