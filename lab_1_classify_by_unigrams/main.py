import json
import re

"""
Lab 1.

Language detection
"""


# pylint:disable=too-many-locals, unused-argument, unused-variable


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

    all_letters = re.sub(r'[\W\d_]', '', text.lower())
    tokens = re.findall(r'(.)', all_letters)
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

    freq_dict = {}
    for character in tokens:
        freq_dict[character] = tokens.count(character) / len(tokens)

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

    profile_dict = {'name': language, 'freq': calculate_frequencies(tokenize(text))}
    return profile_dict


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

    value = 0
    i = 0
    while i < len(actual):
        value += (actual[i] - predicted[i]) ** 2
        i += 1

    MSE = value / i
    return MSE


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

    return calculate_mse(list(profile_to_compare['freq'].values()), list(unknown_profile['freq'].values()))


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

    MSE_profile_1 = compare_profiles(unknown_profile, profile_1)
    MSE_profile_2 = compare_profiles(unknown_profile, profile_2)
    lang_list = [profile_1['name'], profile_2['name']]

    if MSE_profile_1 == MSE_profile_2:
        return sorted(lang_list)[0]
    elif MSE_profile_1 > MSE_profile_2:
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

    with open(f'assets/profiles/{path_to_file}.json', 'r', encoding='utf-8') as file:
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

    profile_freq = profile['freq']
    for key in list(profile_freq.keys()):
        if not len(key) == 1 or not key.isalpha():
            del profile['freq'][key]
            continue
        profile_freq[key] /= profile['n_words'][0]
        profile_freq[key.lower()] = profile_freq.pop(key)
    del profile['n_words']

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

    profiles_list = []
    for path in paths_to_profiles:
        profiles_list.append(preprocess_profile(load_profile(path)))

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

    result_list = []
    for profile in known_profiles:
        result_list.append((profile['name'], compare_profiles(unknown_profile, profile)))
    sorted(sorted(result_list, key=lambda x: x[1]), key=lambda x: x[0])
    return result_list


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """

    for profile in detections:
        print(f'{profile[0]}: MSE {profile[1]:.5f}')
