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
    if not isinstance(text, str) or len(text) == 0:
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
    if not isinstance(tokens, list) or len(tokens) == 0 or None in tokens:
        return None

    freq_dict = {}
    for character in tokens:
        freq_dict[character] = (tokens.count(character) / len(tokens))

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
    if (not isinstance(language, str) or not isinstance(text, str)
            or len(language) == 0 or len(text) == 0):
        return None
    freq_text = calculate_frequencies(tokenize(text))
    if not freq_text:
        return None

    return {'name': language, 'freq': freq_text}


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
    if (not isinstance(predicted, list) or not isinstance(actual, list)
            or len(predicted) == 0 or len(actual) == 0
            or len(predicted) != len(actual)):
        return None

    value = 0
    i = 0
    while i < len(actual):
        value += (actual[i] - predicted[i]) ** 2
        i += 1

    return value / i


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if (len(unknown_profile) == 0 or len(profile_to_compare) == 0
            or len(unknown_profile.keys()) != 2 or len(profile_to_compare.keys()) != 2):
        return None

    profile_to_compare_freq = dict(profile_to_compare['freq'])
    unknown_profile_freq = dict(unknown_profile['freq'])

    for key in profile_to_compare_freq.keys():
        if unknown_profile_freq.get(key) is None:
            unknown_profile_freq.update({key: 0})

    for key in unknown_profile_freq.keys():
        if profile_to_compare_freq.get(key) is None:
            profile_to_compare_freq.update({key: 0})

    sorted_profile_to_compare = list(dict(sorted(profile_to_compare_freq.items(),
                                                 key=lambda x: x[0])).values())
    sorted_unknown_profile = list(dict(sorted(unknown_profile_freq.items(),
                                              key=lambda x: x[0])).values())

    return calculate_mse(sorted_profile_to_compare, sorted_unknown_profile)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict):
        return None
    if (not isinstance(profile_2, dict) or len(unknown_profile) == 0
            or len(profile_1) == 0 or len(profile_2) == 0):
        return None

    mse_profile_1 = compare_profiles(unknown_profile, profile_1)
    mse_profile_2 = compare_profiles(unknown_profile, profile_2)

    if not mse_profile_1 or not mse_profile_2:
        return None

    lang_list = [profile_1['name'], profile_2['name']]

    if mse_profile_1 == mse_profile_2:
        return sorted(lang_list)[0]
    if mse_profile_1 > mse_profile_2:
        return profile_2['name']
    return profile_1['name']


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(path_to_file, str) or len(path_to_file) == 0:
        return None

    with open(path_to_file, 'r', encoding='utf-8') as file:
        load_dict = json.load(file)
    return load_dict


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
    if (not isinstance(profile, dict) or len(profile) == 0 or 'name' not in profile.keys()
            or 'n_words' not in profile.keys() or 'freq' not in profile.keys()):
        return None

    result_freq = {}
    profile_freq = profile['freq']
    list_profile_freq_keys = list(profile_freq.keys())
    for key in list_profile_freq_keys:
        if not len(key) == 1 or profile_freq.get(key) is None:
            continue
        if key.isupper() and profile_freq.get(key.lower()) is not None:
            profile_freq_key = profile_freq[key] / profile['n_words'][0]
            profile_freq_key_lower = profile_freq[key.lower()] / profile['n_words'][0]
            result_freq.update({key.lower(): profile_freq_key + profile_freq_key_lower})
            del profile_freq[key]
            del profile_freq[key.lower()]
        elif key.islower() and profile_freq.get(key.upper()) is not None:
            profile_freq_key = profile_freq[key] / profile['n_words'][0]
            profile_freq_key_upper = profile_freq[key.upper()] / profile['n_words'][0]
            result_freq.update({key.lower(): profile_freq_key + profile_freq_key_upper})
            del profile_freq[key]
            del profile_freq[key.upper()]
        else:
            profile_freq_key = profile_freq[key] / profile['n_words'][0]
            result_freq.update({key.lower(): profile_freq_key})

    del profile['n_words']
    del profile['freq']

    profile.update({'freq': result_freq})

    return profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(paths_to_profiles, list) or len(paths_to_profiles) == 0:
        return None

    profiles_list = []
    for path in paths_to_profiles:
        preprocessed_profile = preprocess_profile(load_profile(path))
        if not preprocessed_profile:
            return None
        profiles_list.append(preprocessed_profile)

    return profiles_list


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
    if (not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list)
            or len(unknown_profile) == 0 or len(known_profiles) == 0):
        return None

    result_list = []
    for profile in known_profiles:
        compared_profiles = compare_profiles(unknown_profile, profile)
        if not compared_profiles:
            return None
        result_list.append((profile['name'], compared_profiles))

    return sorted(sorted(result_list, key=lambda x: x[0]), key=lambda x: x[1], reverse=False)


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(detections, list) or len(detections) == 0:
        return None

    for profile in detections:
        print(f'{profile[0]}: MSE {profile[1]:.5f}')

    return None
