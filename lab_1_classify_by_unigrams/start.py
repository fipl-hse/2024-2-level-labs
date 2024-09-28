"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

import lab_1_classify_by_unigrams.main as funcs


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    unknown_profile = funcs.create_language_profile("unk", unknown_text)

    profile_paths = ['assets/profiles/de.json',
                     'assets/profiles/en.json',
                     'assets/profiles/es.json',
                     'assets/profiles/fr.json',
                     'assets/profiles/it.json',
                     'assets/profiles/ru.json',
                     'assets/profiles/tr.json',]
    profiles = funcs.collect_profiles(profile_paths)

    result = None
    if isinstance(unknown_profile, dict) and isinstance(profiles, list):
        result = funcs.detect_language_advanced(unknown_profile, profiles)
    if (isinstance(result, list) and
        all(isinstance(pair, tuple) for pair in result) and
        all(isinstance(pair[0], str) for pair in result) and
        all(isinstance(pair[1], float) for pair in result)):
        funcs.print_report(result)
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
