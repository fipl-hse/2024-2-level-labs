"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
import json


def tokenize(text: str) -> list[str] | None:
    """
    Split a text into tokens.

    Convert the tokens into lowercase, remove punctuation, digits and other symbols

    Args:
        text (str): A text

    Returns:
        list[str] | None: A list of lower-cased tokens without punctuation

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str):
        return None
    prepared_text = text.lower()
    tokens = []
    for tok in prepared_text:
        if not tok.isalpha():
            continue
        tokens.append(tok)
    return tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list):
        return None
    if not isinstance(tokens[0], str):
        return None
    frequency = {token: tokens.count(token)/len(tokens) for token in set(tokens)}
    return frequency


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Create a language profile.

    Args:
        language (str): A language
        text (str): A text

    Returns:
        dict[str, str | dict[str, float]] | None: A dictionary with two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str) or not isinstance(language, str):
        return None
    freq_dict = calculate_frequencies(tokenize(text))
    if freq_dict is None:
        return None
    return {"name": language, "freq": freq_dict}


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculate mean squared error between predicted and actual values.

    Args:
        predicted (list): A list of predicted values
        actual (list): A list of actual values

    Returns:
        float | None: The score

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if len(actual) != len(predicted):
        return None
    summa = 0.0
    for index, value in enumerate(actual):
        actual_value = value
        predicted_value = float(predicted[index])
        sqw_dif = float((actual_value - predicted_value)**2)
        summa = float(summa + sqw_dif)
    mse = summa/len(actual)
    return mse


def compare_profiles(unknown_profile: dict[str, str | dict[str, float]],
                     profile_to_compare: dict[str, str | dict[str, float]],) -> float | None:
    """
    Compare profiles and calculate the distance using symbols.

    Args:
        unknown_profile (dict[str, str | dict[str, float]]): A dictionary of an unknown profile
        profile_to_compare (dict[str, str | dict[str, float]]): A dictionary of a profile
            to compare the unknown profile to

    Returns:
        float | None: The distance between the profiles

    In case of corrupt input arguments or lack of keys 'name' and
    'freq' in arguments, None is returned
    """
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if len(unknown_profile) != 2 or len(profile_to_compare) != 2:
        return None
    unknown_freq = unknown_profile.get("freq").copy()
    compare_freq = profile_to_compare.get("freq").copy()
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
    if not isinstance(mse, float):
        return None
    mse = round(mse, 4)
    return mse


def detect_language(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_1: dict[str, str | dict[str, float]],
    profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    """
    Detect the language of an unknown profile.

    Args:
        unknown_profile (dict[str, str | dict[str, float]]): A dictionary of a profile
            to determine the language of
        profile_1 (dict[str, str | dict[str, float]]): A dictionary of a known profile
        profile_2 (dict[str, str | dict[str, float]]): A dictionary of a known profile

    Returns:
        str | None: A language

    In case of corrupt input arguments, None is returned
    """
    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if not isinstance(mse_1, float) or not isinstance(mse_2, float):
        return None
    if mse_1 < mse_2:
        language = profile_1.get('name')
        if not isinstance(language, str):
            return None
    elif mse_2 < mse_1:
        language = profile_2.get('name')
        if not isinstance(language, str):
            return None
    else:
        name_1 = profile_1.get("name")
        name_2 = profile_2.get("name")
        if not isinstance(name_1, str) or not isinstance(name_2, str):
            return None
        lang_list = [name_1, name_2]
        lang = sorted(lang_list)
        language = lang[0]
    return language


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, "r", encoding="utf-8") as file_to_read:
        profile_r = json.load(file_to_read)
        profile = dict(profile_r)
    return profile


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocess profile for a loaded language.

    Args:
        profile (dict): A loaded profile

    Returns:
        dict[str, str | dict] | None: A dict with a lower-cased loaded profile
            with relative frequencies without unnecessary n-grams

    In case of corrupt input arguments or lack of keys 'name', 'n_words' and
    'freq' in arguments, None is returned
    """
    if not isinstance(profile, dict) or not isinstance(profile.get('freq'), dict):
        return None
    frequency = profile.get("freq")
    name = profile.get("name")
    if not isinstance(name, str):
        return None
    tokens = list(frequency.keys())
    letters = {}
    n_words_list = profile.get("n_words", int)
    if not isinstance(n_words_list, list):
        return None
    n_words = n_words_list[0]
    for elem in tokens:
        if not isinstance(elem, str):
            return None
        if len(elem) == 1:
            number = frequency.get(elem)
            let = elem.lower()
            if not isinstance(let, str):
                return None
            if let not in letters:
                new_token = {let: number}
                letters.update(new_token)
            else:
                freq_add = letters.get(let)
                sum_freq = number + freq_add
                new_token = {let: sum_freq}
                letters.update(new_token)
    for tok in letters:
        freq_index = float(letters.get(tok) / n_words)
        letters[tok] = freq_index
    return {"name": name, "freq": letters}


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(paths_to_profiles, list):
        return None
    list_of_profiles = []
    for name in paths_to_profiles:
        profile = load_profile(name)
        if not isinstance(profile, dict):
            return None
        correct_profile = preprocess_profile(profile)
        if not isinstance(correct_profile, dict):
            return None
        list_of_profiles.append(correct_profile)
        if not isinstance(list_of_profiles, list):
            return None
    return list_of_profiles


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
    if not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list):
        return None
    results = {}
    for profile in known_profiles:
        mse = compare_profiles(unknown_profile, profile)
        if mse is not None:
            result = {mse: profile.get("name")}
            results.update(result)
    results_s = sorted(results.items())
    data_to_return = []
    for el in results_s:
        new_tok = (el[1], el[0])
        data_to_return.append(new_tok)
    return data_to_return


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    for elem in detections:
        if isinstance(elem, tuple):
            score = elem[1]
            text = f"{elem[0]}: MSE {score:.5f}"
            print(text)
