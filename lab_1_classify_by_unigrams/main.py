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
    split_text = text.lower().split()
    dict_of_tokens = []
    for word in split_text:
        for token in word:
            if token.isalpha():
                dict_of_tokens += token
    return dict_of_tokens


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
    for token in tokens:
        if not isinstance(token, str):
            return None
    freq = {}
    prop = {}
    for token in tokens:
        if token in freq:
            freq[token] += 1
        else:
            freq[token] = 1
    for key, value in freq.items():
        prop[key] = value / len(tokens)
    return prop


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
    frequency = calculate_frequencies(tokenize(text))
    if not frequency:
        return None
    return {
        'name': language,
        'freq': frequency
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
    if not isinstance(predicted, list) or not isinstance(actual,list) or\
            not len(predicted) == len(actual):
        return None
    sum_diff = 0
    for pair_num, actual_num in enumerate(actual):
        sum_diff += (actual_num-predicted[pair_num])**2
    return sum_diff / len(predicted)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict) or not\
            len(unknown_profile) ==2 or not len(profile_to_compare) == 2:
        return None
    unknown_frequency = unknown_profile['freq']
    comparing_frequency = profile_to_compare['freq']
    if (not isinstance(unknown_frequency, dict) or not isinstance(comparing_frequency, dict)):
        return None
    for i in unknown_frequency:
        if i not in comparing_frequency:
            comparing_frequency[i] = 0
    for i in comparing_frequency:
        if i not in unknown_frequency:
            unknown_frequency[i] = 0
    sorted_unknown = dict(sorted(unknown_profile['freq'].items()))
    sorted_compare = dict(sorted(profile_to_compare['freq'].items()))
    sort_comp_list = []
    sort_unkn_list = []
    for i in sorted_unknown.values():
        sort_unkn_list.append(i)
    for i in sorted_compare.values():
        sort_comp_list.append(i)
    return calculate_mse(sort_unkn_list, sort_comp_list)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) or not\
            isinstance(profile_2,dict):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    name1 = str(profile_1['name'])
    name2 = str(profile_2['name'])
    if mse_1 is None or mse_2 is None:
        return None
    if mse_1 < mse_2:
        return name1
    return name2


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
    with open(path_to_file, "r", encoding="utf-8") as read_file:
        read_profile = json.load(read_file)
    if not isinstance(read_profile, dict):
        return None
    return read_profile


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
    if not isinstance(profile, dict)\
            or not all(keys in profile for keys in('freq','name','n_words')):
        return None
    processed_profile = {'name': profile['name'], 'freq': {}}
    frequency_dict = {}
    for token in profile['freq']:
        if isinstance(token, str) and len(token) == 1:
            frequency_dict[token] = profile['freq'][token]
    token_list = list(frequency_dict)
    for letter in token_list:
        if not letter.isupper():
            frequency_dict[letter] = frequency_dict.get(letter,0)\
                                     + frequency_dict.get(letter.upper(),0)
            processed_profile['freq'][letter] = frequency_dict[letter] / profile['n_words'][0]
            if letter.upper() in frequency_dict:
                frequency_dict.pop(letter.upper())
    for i in frequency_dict.items():
        if i[0].isupper():
            processed_profile['freq'][i[0].lower()] = i[1] / profile['n_words'][0]
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
    if not (isinstance(paths_to_profiles,list)
            and all(isinstance(i, str) for i in paths_to_profiles)):
        return None
    collection_of_profiles = []
    for i in paths_to_profiles:
        profile = load_profile(i)
        if not profile:
            return None
        processed_profile = preprocess_profile(profile)
        if not processed_profile:
            return None
        collection_of_profiles.append(processed_profile)
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
    if not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list):
        return None
    list_of_profiles = []
    for profile in known_profiles:
        if not isinstance(compare_profiles(unknown_profile, profile), float):
            return None
        list_of_profiles.append((profile['name'], compare_profiles(unknown_profile, profile)))
    if list_of_profiles:
        list_of_profiles.sort(key=lambda i: (i[1], i[0]))
        return list_of_profiles
    return None



def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(detections, list):
        return None

    for i in detections:
        print(f'{i[0]}: MSE {i[1]:.5f}')
    return None
