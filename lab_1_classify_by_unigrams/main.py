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

    tokens_list = [symbol.lower() for symbol in text if symbol.isalpha()]
    return tokens_list


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(tokens, list) and
            all(isinstance(token, str) for token in tokens)):
        return None

    freq_dict = {}
    for token in tokens:
        freq_dict.setdefault(token, 0)
        freq_dict[token] += 1
    for dict_token in freq_dict:
        freq_dict[dict_token] /= len(tokens)

    return freq_dict


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
    if not (isinstance(language, str) and isinstance(text, str)):
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
    if not (isinstance(predicted, list) and
            isinstance(actual, list) and
            len(predicted) == len(actual) and
            len(predicted) != 0):
        return None

    num_of_diffs = len(predicted)
    sum_of_diffs = 0
    for item_index in range(num_of_diffs):
        sum_of_diffs += (actual[item_index] - predicted[item_index]) ** 2
    mse = sum_of_diffs / num_of_diffs

    return mse


def compare_profiles(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_to_compare: dict[str, str | dict[str, float]],
) -> float | None:
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
    if not (isinstance(unknown_profile, dict) and
            isinstance(profile_to_compare, dict) and
            unknown_profile.get("name") and
            profile_to_compare.get("name") and
            unknown_profile.get("freq") and
            profile_to_compare.get("freq") and
            isinstance(unknown_profile["name"], str) and
            isinstance(profile_to_compare["name"], str) and
            isinstance(unknown_profile["freq"], dict) and
            isinstance(profile_to_compare["freq"], dict)):
        return None

    all_keys = list(set(unknown_profile["freq"]) | set(profile_to_compare["freq"]))
    unknown_freq_list = []
    freq_list_to_compare = []
    for key in all_keys:
        unknown_freq_list.append(unknown_profile["freq"].get(key, 0.0))
        freq_list_to_compare.append(profile_to_compare["freq"].get(key, 0.0))

    return calculate_mse(unknown_freq_list, freq_list_to_compare)


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
    if not (isinstance(unknown_profile, dict) and
            isinstance(profile_1, dict) and
            isinstance(profile_2, dict) and
            isinstance(profile_1["name"], str) and
            isinstance(profile_2["name"], str)):
        return None
    diff_dict = {"diff_unk_1": compare_profiles(unknown_profile, profile_1),
                 "diff_unk_2": compare_profiles(unknown_profile, profile_2)}
    if (isinstance(diff_dict["diff_unk_1"], float) and 
        isinstance(diff_dict["diff_unk_2"], float)):
        diff_dict = dict(sorted(diff_dict.items()))
        if diff_dict["diff_unk_1"] < diff_dict["diff_unk_2"]:
            return profile_1["name"]
        if diff_dict["diff_unk_1"] > diff_dict["diff_unk_2"]:
            return profile_2["name"]
        return list(diff_dict)[0]

    return None


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

    with open(path_to_file, "r", encoding="utf-8") as f:
        profile = json.load(f)
    if not (isinstance(profile, dict) and "name" in profile and "freq" in profile):
        return None

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
    if not (isinstance(profile, dict) and
            "name" in profile and
            "freq" in profile and
            "n_words" in profile):
        return None

    result_freq_dict = {}
    for token in profile["freq"]:
        if isinstance(token, str) and len(token) == 1:
            result_freq_dict.setdefault(token.lower(), 0)
            result_freq_dict[token.lower()] += profile["freq"][token]
    for dict_token in result_freq_dict:
        result_freq_dict[dict_token] /= profile["n_words"][0]

    return {"name": profile["name"], "freq": result_freq_dict}


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(paths_to_profiles, list) and
            all(isinstance(path, str) for path in paths_to_profiles)):
        return None

    result_profiles = []
    for path in paths_to_profiles:
        profile = load_profile(path)
        if not isinstance(profile, dict):
            return None
        profile = preprocess_profile(profile)
        if not isinstance(profile, dict):
            return None
        result_profiles.append(profile)

    return result_profiles


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
    if not (isinstance(unknown_profile, dict) and
            isinstance(known_profiles, list) and
            all(isinstance(prof, dict) for prof in known_profiles)):
        return None

    dist_list = []
    for known_profile in known_profiles:
        diff_unk_known = compare_profiles(unknown_profile, known_profile)
        if not isinstance(diff_unk_known, float):
            return None
        dist_list.append((known_profile["name"], diff_unk_known))

    dist_list.sort(key=lambda a: (a[1], a[0]))
    return dist_list


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(detections, list) and
            all(isinstance(detection, tuple) for detection in detections)):
        return None

    for detection in detections:
        print(f"{detection[0]}: MSE {detection[1]:.5f}")
    return None
