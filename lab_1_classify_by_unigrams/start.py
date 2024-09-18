import main as m

"""
Language detection starter
"""


# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:
    """
    Launches an implementation
    """

    # lang_names_lst = {'en', 'de', 'unknown'}
    # texts_dict = {}
    #
    # for lang_name in lang_names_lst:
    #     file_pattern = f'assets/texts/{lang_name}.txt'
    #     with open(file_pattern, 'r', encoding='utf-8') as file:
    #         read_text = file.read()
    #         texts_dict[f'{lang_name}_text'] = read_text

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
    # result = m.load_profile('ru')

    # Step 8.
    # result = m.preprocess_profile(m.load_profile('tr'))

    # Step 9.
    # result = m.collect_profiles(['es', 'fr', 'it', 'ru', 'tr'])

    # Step 10.
    result = m.detect_language_advanced(m.create_language_profile('unknown', unknown_text),
                                        m.collect_profiles(['es', 'fr', 'it', 'ru', 'tr']))
    print(result)

    # Step 11.
    m.print_report(result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
