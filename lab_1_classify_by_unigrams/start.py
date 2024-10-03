"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
import lab_1_classify_by_unigrams.main as fnc


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
    res1 = fnc.tokenize(en_text)
    print(res1)
    res2 = fnc.calculate_frequencies(res1)
    print(res2)
    en_prof = fnc.create_language_profile('en', en_text)
    print(en_prof)
    assert isinstance(en_prof, dict), 'Dictionary not found'
    de_prof = fnc.create_language_profile('de', de_text)
    assert isinstance(de_prof, dict), 'Dictionary not found'
    trg_prof = fnc.create_language_profile('Unknown', unknown_text)
    assert isinstance(trg_prof, dict), 'Dictionary not found'
    print(fnc.compare_profiles(trg_prof, en_prof))
    print(fnc.compare_profiles(trg_prof, de_prof))
    result = fnc.detect_language(trg_prof, en_prof, de_prof)
    assert result, "Detection result is None"
    print(result)


if __name__ == "__main__":
    main()
