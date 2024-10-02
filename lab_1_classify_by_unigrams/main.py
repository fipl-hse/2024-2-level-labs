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
                return mse

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

