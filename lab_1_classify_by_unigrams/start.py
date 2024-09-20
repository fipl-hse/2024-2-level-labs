"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
def main() -> None:
    """
    Launches an implementation
    """
    from main import tokenize, create_language_profile

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        tokenize(en_text)
        create_language_profile('en', en_text)

    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        tokenize(de_text)
        create_language_profile('de', de_text)

    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        tokenize(unknown_text)
        create_language_profile('unknown', unknown_text)

    extra_profiles = ["lab_1_classify_by_unigrams\\assets\\profiles\\de.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\en.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\es.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\fr.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\it.json",
                      "lab_1_classify_by_unigrams\\assets\\profiles\\ru.json"
                      "lab_1_classify_by_unigrams\\assets\\profiles\\tr.json"]

    result = None
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()

