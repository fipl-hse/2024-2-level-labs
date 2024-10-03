"""
Lab 1.

Language detection
"""
import json

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
    if not isinstance(text, str) or len(text) < 1:
        return None
    tokens = []
    for symbol in text.lower().strip():
        if not symbol.isalpha():
            continue
        tokens += symbol
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
    if tokens is None or not isinstance(tokens, list):
        return None

    freq_dict = {}
    new_tokens = []

    for token in tokens:
        if not isinstance(token, str) or len(token) != 1:
            return None
        new_tokens.append(token)

    all_tokens = len(new_tokens)
    if all_tokens == 0:
        return None

    for token in new_tokens:
        token_freq = new_tokens.count(token) / all_tokens
        freq_dict[token] = token_freq

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
    if (language is None or not isinstance(language, str)
            or text is None or not isinstance(text, str)):
        return None

    tokenized_text = tokenize(text)
    if tokenized_text is None:
        return None

    frequencies_values = calculate_frequencies(tokenized_text)
    if frequencies_values is None:
        return None

    return {"name": language, "freq": frequencies_values}


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
    number_of_values = len(actual)
    mse = float(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / number_of_values)
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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if not all(key in unknown_profile for key in ['name', 'freq']):
        return None
    if not all(key in profile_to_compare for key in ['name', 'freq']):
        return None

    name1 = unknown_profile.get('name')
    freq1 = unknown_profile.get('freq')
    name2 = profile_to_compare.get('name')
    freq2 = profile_to_compare.get('freq')

    if not (isinstance(name1, str) or isinstance(freq1, dict)
            or isinstance(name2, str) or isinstance(freq2, dict)):
        return None

    all_tokens_set = set(freq1.keys()) | set(freq2.keys())
    actual_values, predicted_values = [], []

    for token in all_tokens_set:

        actual_value = freq1.get(token, 0.0)
        actual_values.append(actual_value)

        predicted_value = freq2.get(token, 0.0)
        predicted_values.append(predicted_value)

    mse = calculate_mse(actual_values, predicted_values)

    if mse is None:
        return None

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
    if unknown_profile is None or profile_1 is None or profile_2 is None:
        return None
    if (not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None

    name1 = profile_1.get('name')
    name2 = profile_2.get('name')
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    result1 = name1
    result2 = name2

    if mse_1 is None or mse_2 is None:
        return None

    if mse_1 < mse_2:
        return result1
    if mse_1 == mse_2:
        profile_sorted_list = sorted([result1, result2])
        return str(profile_sorted_list[0])

    return result2


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
    with open(path_to_file, 'r', encoding='UTF-8') as file:
        profile = json.load(file)
    if not isinstance(profile, dict):
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
    if profile is None or not isinstance(profile, dict):
        return None
    if (not all(key in profile for key in ['name', 'freq', 'n_words'])
            or not isinstance(profile['name'], str)
            or not isinstance(profile['freq'], dict)
            or not isinstance(profile['n_words'], list)):
        return None

    name = profile['name']
    freq = profile['freq']
    n_words = profile['n_words']

    total_number = n_words[0]
    if not isinstance(total_number, int) or total_number < 1:
        return None

    freq_dict = {}
    for unigram, value in freq.items():
        if len(unigram) != 1:
            continue
        if not isinstance(unigram, str) or not isinstance(value, int):
            return None
        letter = unigram.lower()
        if letter.isalpha() or unigram == '²' or letter == 'i̇':
            if freq_dict.get(letter) is None:
                freq_dict[letter] = 0.0
            freq_dict[letter] += value / total_number

    return {'name': name, 'freq': freq_dict}


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if (not isinstance(paths_to_profiles, list)
            or not all(isinstance(path, str) for path in paths_to_profiles)):
        return None

    collected_profiles = []

    for path in paths_to_profiles:
        profile = load_profile(path)
        if profile is None:
            return None
        processed_profile = preprocess_profile(profile)
        if processed_profile is None or not isinstance(processed_profile, dict):
            return None
        collected_profiles.append(processed_profile)

    return collected_profiles


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
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
    if (unknown_profile is None or known_profiles is None
            or not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list)):
        return None
    results_not_sorted = {}
    for profile in known_profiles:
        if (not isinstance(profile, dict)
                or not all(key in profile for key in ['name', 'freq'])):
            return None
        mse = compare_profiles(unknown_profile, profile)
        if mse is None:
            return None
        lang = profile['name']
        if lang is None:
            return None
        result = {lang: mse}
        results_not_sorted.update(result)

    results_lst = list(results_not_sorted.items())
    results_sorted = sorted(results_lst, key=lambda x: (x[1], x[0]))
    return results_sorted


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if detections is None or not isinstance(detections, list):
        return
    for element in detections:
        if not isinstance(element, tuple):
            return
        value = element[1]
        print(f'{element[0]}: MSE {value:.5f}')
