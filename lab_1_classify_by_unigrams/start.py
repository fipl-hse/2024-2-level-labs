"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
import lab_1_classify_by_unigrams.main as m


def main() -> None:
    """
    Launches an implementation
    """

    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    result = m.detect_language_advanced(m.create_language_profile('unknown', unknown_text),
                                        m.collect_profiles(['assets/profiles/es.json',
                                                            'assets/profiles/fr.json',
                                                            'assets/profiles/it.json',
                                                            'assets/profiles/ru.json',
                                                            'assets/profiles/tr.json']))
    print(result)
    m.print_report(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
