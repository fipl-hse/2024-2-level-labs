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
    tokens = []
    for symbol in text:
        if symbol.isalpha():
            tokens.append(symbol.lower())
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
    is_corrupt_input = not (isinstance(tokens, list) and all(isinstance(token, str) for token in tokens))
    if is_corrupt_input:
        return None

    frequencies_of_letters = {}
    for token in tokens:
        if frequencies_of_letters.get(token) == None:
            frequencies_of_letters[token] = 0
        frequencies_of_letters[token] += 1
    for token in frequencies_of_letters:
        frequencies_of_letters[token] /= len(tokens)
    return frequencies_of_letters


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
    is_corrupt_input = (isinstance(language, str)) and (isinstance(text, str))
    if not is_corrupt_input:
        return None
    tokens = tokenize(text)
    freq_dict = calculate_frequencies(tokens)
    language_profile = {
        "name": language,
        "freq": freq_dict,
    }
    return language_profile


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
    is_corrupt_input = (isinstance(predicted, list)) and (isinstance(actual, list))
    if not is_corrupt_input:
        return None
    mse = 0
    n = len(predicted)
    for index in range(n):
        mse += (actual[index] - predicted[index]) ** 2
    mse /= n
    return mse

def is_valid_profile(profile):
    if profile.get("name") == None or profile.get("freq") == None:
        return False
    if not isinstance(profile.get("name"), str):
        return False
    if not isinstance(profile.get("freq"), dict):
        return False
    for letter, freq in profile["freq"].items():
        if not isinstance(letter, str):
            return False
        if not isinstance(freq, float):
            return False
    return True

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
    if not (is_valid_profile(unknown_profile) and is_valid_profile(profile_to_compare)):
        return None

    unknown_tokens = set(unknown_profile["freq"])
    tokens_to_compare = set(profile_to_compare["freq"])
    union = unknown_tokens | tokens_to_compare
    values_unknown_tokens = []
    values_tokens_to_compare = []
    for letter in union:
        if unknown_profile["freq"].get(letter) == None:
            values_unknown_tokens.append(0)
        else:
            values_unknown_tokens.append(unknown_profile["freq"].get(letter))
        if profile_to_compare["freq"].get(letter) == None:
            values_tokens_to_compare.append(0)
        else:
            values_tokens_to_compare.append(profile_to_compare["freq"].get(letter))
    result = calculate_mse(values_unknown_tokens, values_tokens_to_compare)
    return result




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
    if not (is_valid_profile(unknown_profile) and is_valid_profile(profile_1) and is_valid_profile(profile_2)):
        return None

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 < mse_2:
        return profile_1["name"]
    if mse_2 < mse_1:
        return profile_2["name"]
    else:
        return sorted([profile_1["name"], profile_2["name"]])[0]


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    is_corrupt_input = isinstance(path_to_file, str)
    if not is_corrupt_input:
        return None
    with open(path_to_file, encoding='utf-8') as f:
        profile = json.load(f)
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
    is_valid_input = (isinstance(profile, dict) and profile.get("name")
                      and profile.get("n_words") and profile.get("freq"))
    if not is_valid_input:
        return None

    new_profile = {
        "name": profile["name"],
        "freq": {}
    }
    for unigram in profile["freq"]:
        if len(unigram) == 1 and unigram.isalpha() and unigram != "가":
            new_profile["freq"][unigram.lower()] = profile["freq"][unigram] / profile["n_words"][0]
    return new_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    is_corrupt_input = isinstance(paths_to_profiles, list)
    if not is_corrupt_input:
        return None
    collection_of_profiles = []
    for profile in paths_to_profiles:
        collection_of_profiles.append(preprocess_profile(load_profile(profile)))
    return collection_of_profiles


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
    if not (is_valid_profile(unknown_profile) and isinstance(known_profiles, list)):
        return None

    sorted_list = []
    for profile in known_profiles:
        mse = compare_profiles(unknown_profile, profile)
        tuple_lang_distance = (profile["name"], mse)
        sorted_list.append(tuple_lang_distance)
    sorted_list.sort(key=lambda profile: (profile[1], profile[0]))
    return sorted_list


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(detections, list):
        return None
    for detection in detections:
        if not isinstance(detection, tuple):
            return None
        if not isinstance(detection[0], str):
            return None
        if not isinstance(detection[1], float):
            return None
    for detection in detections:
        print(f"{detection[0]}: MSE {detection[1]:.5f}")
