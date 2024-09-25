"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
from main import tokenize, calculate_frequencies, calculate_mse, create_language_profile, compare_profiles, \
    detect_language


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
    res1 = tokenize(en_text)
    print(res1)
    res2 = calculate_frequencies(res1)
    print(res2)
    en_prof = create_language_profile('en', en_text)
    print(en_prof)
    de_prof = create_language_profile('de', de_text)
    trg_prof = create_language_profile('Unknown', unknown_text)
    print(compare_profiles(trg_prof, en_prof))
    print(compare_profiles(trg_prof, de_prof))
    print(detect_language(trg_prof, en_prof, de_prof))

    # result = None
    # assert result, "Detection result is None"


if __name__ == "__main__":
    main()
