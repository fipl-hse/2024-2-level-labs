"""
Language detection starter
"""
from lab_1_classify_by_unigrams import main as func


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
    result = func.tokenize(en_text)
    print(result)
    print(func.calculate_frequencies(result))
    print(func.create_language_profile('en', en_text))
    en_profile = func.create_language_profile('en', en_text)
    du_profile = func.create_language_profile('de', de_text)
    un_profile = func.create_language_profile('un', unknown_text)
    if en_profile is None or du_profile is None or un_profile is None:
        return
    det_result = func.detect_language(un_profile, du_profile, en_profile)

    if det_result is None:
        print("failed")
    else:
        print(det_result)
    profiles_paths = ["assets/profiles/de.json",
                      "assets/profiles/en.json",
                      "assets/profiles/es.json",
                      "assets/profiles/fr.json",
                      "assets/profiles/it.json",
                      "assets/profiles/ru.json",
                      "assets/profiles/tr.json"]
    known_profiles = func.collect_profiles(profiles_paths)
    if known_profiles is None:
        return
    advanced_detection_result = func.detect_language_advanced(un_profile, known_profiles)
    func.print_report(advanced_detection_result)

    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
