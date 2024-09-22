import lab_1_classify_by_unigrams.main as m

"""
Language detection starter
"""


# pylint:disable=too-many-locals, unused-argument, unused-variable


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

    # Step 1.
    # result = m.tokenize(en_text)

    # Step 2.
    # result = m.calculate_frequencies(m.tokenize(de_text))

    # Step 3.
    # result = m.create_language_profile('de', de_text)

    # Step 4.
    # result = m.calculate_mse(list(m.create_language_profile('de', de_text)['freq'].values()),
    #                          list(m.create_language_profile('unknown', unknown_text)['freq'].values()))

    # Step 5.
    # result = m.compare_profiles(m.create_language_profile('unknown', unknown_text),
    #                             m.create_language_profile('de', de_text))

    # Step 6.
    # result = m.detect_language(m.create_language_profile('unknown', unknown_text),
    #                            m.create_language_profile('de', de_text),
    #                            m.create_language_profile('en', en_text))

    # Step 7.
    # result = m.load_profile('assets/profiles/ru.json')

    # Step 8.
    # result = m.preprocess_profile(m.load_profile('tr'))

    # Step 9.
    # result = m.collect_profiles(['assets/profiles/es.json', 'assets/profiles/fr.json', 'assets/profiles/it.json',
    #                              'assets/profiles/ru.json', 'assets/profiles/tr.json'])

    # Step 10.
    result = m.detect_language_advanced(m.create_language_profile('unknown', unknown_text),
                                        m.collect_profiles(['assets/profiles/es.json', 'assets/profiles/fr.json',
                                                            'assets/profiles/it.json', 'assets/profiles/ru.json',
                                                            'assets/profiles/tr.json']))
    print(result)

    # Step 11.
    m.print_report(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
