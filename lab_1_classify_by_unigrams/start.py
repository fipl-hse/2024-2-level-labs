from main import calculate_frequencies, create_language_profile, detect_language, tokenize

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
    result = tokenize(en_text)
    print(result)
    print(calculate_frequencies(result))
    print(create_language_profile('en', en_text))
    profile_en = create_language_profile('en', en_text)
    profile_de = create_language_profile('de', de_text)
    profile_un = create_language_profile('un', unknown_text)
    if profile_un and profile_en and profile_de:
        print(detect_language(profile_unk, profile_en, profile_de))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
