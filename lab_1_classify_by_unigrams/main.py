"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable


import copy
import json

def tokenize(text: str) -> list[str] | None:
    """
    Split a text into tokens.

    Convert the tokens into lowercase, remove punctuation, digits and other symbols.

    Args:
        text (str): A text

    Returns:
        list[str] | None: A list of lower-cased tokens without punctuation.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str):
        return None

    
    text = text.lower()

    list_of_tokens = []
    for token in text.lower():
        if token.isalpha():
            list_of_tokens.append(token)
    return list_of_tokens



def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return None

    frequency = {}
    for token in set(tokens):
        frequency[token] = tokens.count(token) / len(tokens)
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
    if not isinstance(language, str) or not isinstance(text, str):
        return None

    frequency_dict = calculate_frequencies(tokenize(text))
    if not frequency_dict:
        return None

    return {"name": language, "freq": frequency_dict}




def calculate_mse(predicted: list[float], actual: list[float]) -> float | None:
    """
    Calculate mean squared error between predicted and actual values.

    Args:
        predicted (list): A list of predicted values
        actual (list): A list of actual values

    Returns:
        float | None: The score

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(predicted, list) and isinstance(actual, list)):
        return None
    sum_diff = 0
    for i, value in enumerate(actual):
        difference_between_values = (value - predicted[i]) ** 2
        sum_diff += difference_between_values
    return sum_diff / len(predicted)




def compare_profiles(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:
    """
    Compare profiles and calculate the distance using symbols.

    Args:
        unknown_profile (dict[str, str | dict[str, float]]): A dictionary of an unknown profile
        profile_to_compare (dict[str, str | dict[str, float]]): A dictionary of a profile
            to compare the unknown profile to

    Returns:
        float | None: The distance between the profiles

    In case of corrupt input arguments or lack of keys 'name' and 'freq' in arguments, None is returned
    """
    if not all(isinstance(profile, dict) for profile in [unknown_profile, profile_to_compare]):
        return None
    if ('name' or 'freq') not in unknown_profile or ('name' or 'freq') not in profile_to_compare:
        return None
    copy_unk_profile = copy.deepcopy(unknown_profile)
    if (not isinstance(copy_unk_profile['freq'], dict)
            or not isinstance(profile_to_compare['freq'], dict)):
        return None
    for letter in copy_unk_profile['freq']:
        if letter not in profile_to_compare['freq']:
            profile_to_compare['freq'][letter] = 0
    for letter in profile_to_compare['freq']:
        if letter not in copy_unk_profile['freq']:
            copy_unk_profile['freq'][letter] = 0
    sort_unk = dict(sorted(copy_unk_profile['freq'].items()))
    sort_comp = dict(sorted(profile_to_compare['freq'].items()))
    comp_values_lst = []
    for freq in sort_comp.values():
        comp_values_lst.append(freq)
    unk_values_lst = []
    for freq in sort_unk.values():
        unk_values_lst.append(freq)
    return calculate_mse(comp_values_lst, unk_values_lst)

def detect_language(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_1: dict[str, str | dict[str, float]],
    profile_2: dict[str, str | dict[str, float]]
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
    if (not isinstance(profile_1, dict) or not isinstance(profile_2, dict)
            or not isinstance(unknown_profile, dict)):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 is None or mse_2 is None:
        return None
    if not isinstance(profile_1['name'], str) or not isinstance(profile_2['name'], str):
        return None
    if mse_1 < mse_2:
        return profile_1['name']
    if mse_1 > mse_2:
        return profile_2['name']
    return sorted([profile_1['name'], profile_2['name']])[0]


