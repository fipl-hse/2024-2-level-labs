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
    if isinstance(text, str) and len(text) > 0:
        tokens = []
        for symbol in text.lower().strip():
            if symbol.isalpha():
                tokens += symbol
        return tokens

    return None


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
        if isinstance(token, str) and len(token) == 1:
            new_tokens.append(token)
        else:
            return None

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
    if (language is not None and isinstance(language, str)
            and text is not None and isinstance(text, str)):
        tokenized_text = tokenize(text)
        frequencies_values = calculate_frequencies(tokenized_text)

        if frequencies_values is not None:
            return {"name": language, "freq": frequencies_values}

    return None


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
    if isinstance(predicted, list) and isinstance(actual, list):
        if len(actual) == len(predicted):
            n = len(actual)
            mse = float(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / n)
            return mse
    return None


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
    if isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict):
        name1 = unknown_profile.get('name')
        freq1 = unknown_profile.get('freq')
        name2 = profile_to_compare.get('name')
        freq2 = profile_to_compare.get('freq')

        if (isinstance(name1, str) and isinstance(freq1, dict)
                and isinstance(name2, str) and isinstance(freq2, dict)):
            all_tokens_set = set(freq1.keys()) | set(freq2.keys())
            actual_values, predicted_values = [], []

            for token in all_tokens_set:

                actual_value = freq1.get(token, 0.0)
                actual_values.append(actual_value)

                predicted_value = freq2.get(token, 0.0)
                predicted_values.append(predicted_value)

            mse = calculate_mse(actual_values, predicted_values)

            if mse is not None:
                final_mse = round(mse, 3)
                return final_mse

            return None
        return None
    return None


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
    if unknown_profile is not None and profile_1 is not None and profile_2 is not None:
        if (isinstance(unknown_profile, dict)
                and isinstance(profile_1, dict) and isinstance(profile_2, dict)):

            name1 = profile_1.get('name')
            name2 = profile_2.get('name')
            mse_1 = compare_profiles(unknown_profile, profile_1)
            mse_2 = compare_profiles(unknown_profile, profile_2)
            result1 = str(name1)
            result2 = str(name2)

            if mse_1 is not None and mse_2 is not None:
                if mse_1 < mse_2:
                    return result1
                if mse_1 == mse_2:
                    profile_sorted_list = sorted([result1, result2])
                    return str(profile_sorted_list[0])
                return result2

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
    if isinstance(path_to_file, str):
        with open(path_to_file, 'r', encoding='UTF-8') as file:
            profile = json.load(file)
        if isinstance(profile, dict):
            return profile

    return None


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

    if not (isinstance(profile, dict)
            and all(key in profile for key in ['name', 'freq', 'n_words'])
            and isinstance(profile['name'], str) and isinstance('freq', dict)
            and isinstance('n_words', list)):
        return None

    name: str = profile['name']
    freq: dict = profile['freq']
    n_words: list = profile['n_words']

    total_number: int = n_words[0]
    if total_number < 1:
        return None

    freq_dict = {}
    for k, v in freq.items():
        if len(k) == 1 and isinstance(k, str) and isinstance(v, int):
            freq_dict[k.lower()] = v / total_number

    processed_profile = {'name': name, 'freq': freq_dict}
    return processed_profile


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

    collected_profiles = []

    for path in paths_to_profiles:
        new_profile = preprocess_profile(load_profile(path))
        if new_profile is None:
            return None
        collected_profiles.append(new_profile)

    return collected_profiles


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
    if (unknown_profile is None or not isinstance(unknown_profile, dict)
            or known_profiles is None or not isinstance(known_profiles, list)):
        return None

    for profile in known_profiles:
        if not isinstance(profile, dict) or not all(key in profile for key in ['name', 'freq']):
            return None

    results_not_sorted = {}
    for profile in known_profiles:
        mse = compare_profiles(unknown_profile, profile)
        if mse is not None:
            result = {profile.get('name'): mse}
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
    for element in detections:
        if isinstance(element, tuple):
            value = element[1]
            print(f'{element[0]}: MSE {value:.5f}')
