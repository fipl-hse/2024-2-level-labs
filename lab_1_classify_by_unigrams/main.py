"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable


def tokenize(text):
    if type(text) is str:
        text = text.lower()
        tokens = []
        for i in text:
            if i.isalpha():
                tokens.append(i)
        return tokens


def calculate_frequencies(tokens):
    if type(tokens) is list:
        if type(tokens[0]) is str:
            frequency = {}
            tokens_number = len(tokens)
            for letter in tokens:
                if type(letter) is str:
                    letter_number = tokens.count(letter)
                    letter_index = letter_number/tokens_number
                    letter_dict = {letter: letter_index}
                    frequency.update(letter_dict)
            return frequency


def create_language_profile(language, text) -> dict[str, str | dict[str, float]] | None:
    if type(text) is str and type(language) is str:
        freq_dict = calculate_frequencies(tokenize(text))
        language_profile = {"name": language,
                            "freq": freq_dict
                            }
        return language_profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    if type(predicted) is list and type(actual) is list:
        if len(actual) == len(predicted):
            summa = 0
            n = len(actual)
            for i in range(0, n):
                a = float(actual[i])
                p = float(predicted[i])
                sqw_dif = (a - p)**2
                summa += sqw_dif
            mse = summa/n
            return mse


def compare_profiles(unknown_profile: dict[str, str | dict[str, float]],
                     profile_to_compare: dict[str, str | dict[str, float]],) -> float | None:
    if type(unknown_profile) is dict and type(profile_to_compare) is dict:
        if len(unknown_profile) == 2 and len(profile_to_compare) == 2:
            unknown_freq = unknown_profile.get("freq")
            compare_freq = profile_to_compare.get("freq")
            for tok in unknown_freq:
                if tok not in compare_freq:
                    new_tok = {tok: 0.0}
                    compare_freq.update(new_tok)
            compare_freq_l = sorted(compare_freq.items())
            compare_freq_new = {}
            for elem in compare_freq_l:
                if len(elem) == 2:
                    token_for_new = {elem[0]: elem[1]}
                    compare_freq_new.update(token_for_new)

            for tok in compare_freq:
                if tok not in unknown_freq:
                    new_tok = {tok: 0.0}
                    unknown_freq.update(new_tok)
            unknown_freq_l = sorted(unknown_freq.items())
            unknown_freq_new = {}
            for elem in unknown_freq_l:
                if len(elem) == 2:
                    token_for_new = {elem[0]: elem[1]}
                    unknown_freq_new.update(token_for_new)
            profile_pr = list(compare_freq_new.values())
            profile_act = list(unknown_freq_new.values())
            mse = calculate_mse(profile_pr, profile_act)
            mse = round(mse, 3)
            return mse


def detect_language(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_1: dict[str, str | dict[str, float]],
    profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    if type(unknown_profile) is dict and type(profile_1) is dict and type(profile_2) is dict:
        mse_1 = compare_profiles(unknown_profile, profile_1)
        mse_2 = compare_profiles(unknown_profile, profile_2)
        if mse_1 < mse_2:
            language = profile_1.get('name')
        elif mse_2 < mse_1:
            language = profile_2.get('name')
        else:
            lang_list = [profile_1.get('name'), profile_2.get('name')]
            language = sorted(lang_list)[0]
        return language


def load_profile(path_to_file: str) -> dict | None:
    if type(path_to_file) is str:
        import json
        with open(path_to_file, "r", encoding="utf-8") as file_to_read:
            profile_r = json.load(file_to_read)
            profile = dict(profile_r)
        return profile


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    if type(profile) is dict:
        new_profile = {}
        name = {"name": profile.get("name")}
        new_profile.update(name)
        frequency = profile.get("freq")
        new_freq = {}
        for tok in frequency.keys():
            if len(tok) == 1 and tok.isalpha:
                freq_index = frequency.get(tok)
                tok_correct = {tok:}


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """


def detect_language_advanced(
    unknown_profile: dict[str, str | dict[str, float]], known_profiles: list
) -> list | None:
    """
    Detect the language of an unknown profile.

    Args:
        unknown_profile (dict[str, str | dict[str, float]]): A dictionary of a profile
            to determine the language of
        known_profiles (list): A list of known profiles

    Returns:
        list | None: A sorted list of tuples containing a language and a distance

    In case of corrupt input arguments, None is returned
    """


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
