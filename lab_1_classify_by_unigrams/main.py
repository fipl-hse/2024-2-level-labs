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
    text_tokens = []
    for i in text:
        if i.isalpha():
            text_tokens += i.lower()
    return text_tokens


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)):
        return None
    frequencies = {}
    for l in tokens:
        if l in frequencies:
            continue
        frequencies[l] = tokens.count(l)/len(tokens)
    return frequencies


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
    frequencies = calculate_frequencies(tokenize(text))
    if not (isinstance(language, str) and isinstance(frequencies, dict)):
        return None
    return {
        'name': language,
        'freq': frequencies
    }


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
    if len(predicted) != len(actual):
        return None
    if not (all(isinstance(p, (float, int)) for p in predicted) and all(isinstance(a, (float, int))
                                                                        for a in actual)):
        return None
    mse = 0
    quantity = len(actual)
    for i in range(quantity):
        mse += (actual[i] - predicted[i]) ** 2
    return round((mse / len(actual)), 4)


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
    if not ("name" in unknown_profile and "freq" in unknown_profile and "name" in profile_to_compare
            and "freq" in profile_to_compare):
        return None
    unknown_freq = []
    profile_freq = []
    for k, v in unknown_profile['freq'].items():
        if k not in profile_to_compare['freq'].keys():
            unknown_freq.append(v)
            profile_freq.append(0.0)
        else:
            for k_2, v_2 in profile_to_compare['freq'].items():
                if k_2 == k:
                    unknown_freq.append(v)
                    profile_freq.append(v_2)

    for k, v in profile_to_compare['freq'].items():
        if k not in unknown_profile['freq'].keys():
            unknown_freq.append(0.0)
            profile_freq.append(v)
    mse = calculate_mse(unknown_freq, profile_freq)
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
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 is None or mse_2 is None:
        return None
    if mse_1 < mse_2:
        return profile_1.get('name')
    if mse_1 > mse_2:
        return profile_2.get('name')
    languages = [profile_1.get('name'), profile_2.get('name')]
    languages.sort()
    return languages[0]


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(path_to_file,str):
        return None
    with open(path_to_file, 'r', encoding="utf-8") as file_to_read:
        profile = json.load(file_to_read)
    if not isinstance(profile, dict) or profile is None:
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
    profile_processed = {}
    quantity = 0
    if not ("name" in profile and "freq" in profile and "n_words" in profile):
        return None
    for k, v in profile.items():
        if k == 'name':
            profile_processed[k] = v.lower()
            continue
        if k == 'n_words':
            quantity += v[0]
            continue
        if k == 'freq':
            profile_processed[k] = {}
            for k_2, v_2 in v.items():
                if len(k_2) == 1:
                    if k_2.lower() not in profile_processed[k]:
                        profile_processed[k][k_2.lower()] = v_2
                    elif k_2.lower() in profile_processed[k]:
                        profile_processed[k][k_2.lower()] += v_2
    for k, v in profile_processed['freq'].items():
        profile_processed['freq'][k] = v / quantity
    return profile_processed


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(paths_to_profiles, list) and all(isinstance(p, str)
                                                       for p in paths_to_profiles)):
        return None
    all_profiles = []
    for path in paths_to_profiles:
        all_profiles.append(preprocess_profile(load_profile(path)))
    return all_profiles


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
    distance = []
    if not (isinstance(unknown_profile, dict) and isinstance(known_profiles, list) and
            all(isinstance(elem, dict) for elem in known_profiles)):
        return None
    for profile in known_profiles:
        distance.append((profile['name'], compare_profiles(unknown_profile, profile)))
    return sorted(distance, key=lambda i:i[1])


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if not all(isinstance(elem, tuple) for elem in detections):
        return None
    for elem in detections:
        print(f'{elem[0]}: MSE {elem[1]:.5f}')
    return None
