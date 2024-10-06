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

    Convert the tokens into lowercase, remove punctuation, digits and other symbols

    Args:
        text (str): A text

    Returns:
        list[str] | None: A list of lower-cased tokens without punctuation

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(text, str):
        return None
    tokenized_list = []
    for token in text.lower():
        if token.isalpha():
            tokenized_list.append(token)
    return tokenized_list


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
    frequency_dict = {}
    for token in set(tokens):
        frequency_dict[token] = tokens.count(token) / len(tokens)
    return frequency_dict

    if not isinstance(tokens, list):
        return None

    token_freq = {}

    for token in tokens:
        if not isinstance(token, str):
            return None
        if len(token) > 1:
            return None
        if token not in token_freq:
            token_freq[token] = 1.0
        else:
            token_freq[token] += 1.0

    total_tokens = len(tokens)

    for key, value in token_freq.items():
        token_freq[key] = value / total_tokens


    return token_freq





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
    return {'name': language,
            'freq': frequency_dict}


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
            or not len(predicted) == len(actual)):
        return None
    sum_diff = 0
    for i, value in enumerate(actual):
        difference_between_values = (value - predicted[i]) ** 2
        sum_diff += difference_between_values
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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
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

    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_to_compare, dict)):
        return None
    if ("name" not in unknown_profile
            or "freq" not in unknown_profile
            or "name" not in profile_to_compare
            or "freq" not in profile_to_compare):
        return None
    if (not isinstance(profile_to_compare["name"], str)
            or not isinstance(profile_to_compare["freq"], dict)):
        return None

    all_keys = list(set(unknown_profile['freq']) | set(profile_to_compare['freq']))
    unknown_freq_list = []
    freq_list_to_compare = []

    for key in all_keys:
        if not key in unknown_profile['freq'] and isinstance(unknown_profile['freq'], dict):
            unknown_freq_list.append(0.0)
        if key in unknown_profile['freq'] and isinstance(unknown_profile['freq'], dict):
            freq_of_key = unknown_profile['freq'][str(key)]
            unknown_freq_list.append(float(freq_of_key))
        if not key in profile_to_compare['freq'] and isinstance(profile_to_compare['freq'], dict):
            freq_list_to_compare.append(0.0)
        if key in profile_to_compare['freq'] and isinstance(profile_to_compare['freq'], dict):
            freq_of_key = profile_to_compare['freq'][str(key)]
            freq_list_to_compare.append(float(freq_of_key))

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
    with open(path_to_file, 'r', encoding='utf-8') as file_to_profile:
        profile = json.load(file_to_profile)
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
    if not isinstance(profile, dict) or not all(k in profile for k in ('freq', 'name', 'n_words')):
        return None
    processed_profile = {'name': profile['name'],
                         'freq': {}}
    freq_dict = {}
    for token in profile['freq']:
        if isinstance(token, str) and len(token) == 1:
            freq_dict[token] = profile['freq'][token]
    letters_list = list(freq_dict)
    for letter in letters_list:
        if not letter.isupper():
            freq_dict[letter] = freq_dict.get(letter, 0) + freq_dict.get(letter.upper(), 0)
            processed_profile['freq'][letter] = freq_dict[letter] / profile['n_words'][0]
            if letter.upper() in freq_dict:
                freq_dict.pop(letter.upper())
    for i in freq_dict.items():
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
    if not (isinstance(paths_to_profiles, list)
            and all(isinstance(path, str) for path in paths_to_profiles)):
        return None
    profiles_collection = []
    for path in paths_to_profiles:
        profile = load_profile(path)
        if not profile:
            return None
        pre_profile = preprocess_profile(profile)
        if not pre_profile:
            return None
        profiles_collection.append(pre_profile)
    return profiles_collection


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
    profiles_list = []
    for profile in known_profiles:
        profiles_list.append((profile['name'], compare_profiles(unknown_profile, profile)))
    if profiles_list:
        profiles_list.sort(key=lambda x: (x[-1], x[0]))
        return profiles_list
    return None


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    for detection in detections:
        print(f'{detection[0]}: MSE {detection[-1]:.5f}')
