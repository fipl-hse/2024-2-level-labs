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
    wrong_cases = list("!?/|&><%.,';:\"#\\@()*-+=`~ 1234567890")
    return [symbol for symbol in text.lower() if symbol not in wrong_cases]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or not all(isinstance(s, str) for s in tokens):
        return None
    frequency = {}
    for letter in tokens:
        if letter.isalpha():
            counter = tokens.count(letter) / len(tokens)
            frequency[letter] = counter
    return frequency


def create_language_profile(language: str, text: str) -> \
        (dict[str, str | dict[str, float] | None] | None):
    """
    Create a language profile.

    Args:
        language (str): A language
        text (str): A text

    Returns:
        dict[str, str | dict[str, float]] | None: A dictionary with two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    if ((not isinstance(language, str)) or
            (not isinstance(text, str)) or (tokenize(text) is None) or
            (calculate_frequencies(tokenize(text)) is None)):
        return None
    profile = {
        "name": language,
        "freq": calculate_frequencies(tokenize(text))
    }
    return profile


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
    if not isinstance(predicted, list):
        return None
    if not isinstance(actual, list):
        return None
    if len(predicted) != len(actual):
        return None
    difference = 0
    length = len(predicted)
    for index in range(length):
        difference += (actual[index] - predicted[index]) ** 2
    mse = difference / float(length)
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
    if ((not isinstance(unknown_profile, dict)) or
            (not isinstance(profile_to_compare, dict))):
        return None
    if (len(profile_to_compare.keys()) != 2 or
            (not all(k in profile_to_compare for k in ('freq', 'name')))):
        return None
    if (len(unknown_profile.keys()) != 2 or
            (not all(k in unknown_profile for k in ('freq', 'name')))):
        return None
    unk_profile = copy.deepcopy(unknown_profile)
    unknown_text = unk_profile["freq"]
    comparing_text = profile_to_compare["freq"]
    keys_for_unk = set(unknown_text.keys())
    keys_for_comparing = set(comparing_text.keys())
    if keys_for_unk.intersection(keys_for_comparing) == {}:
        return None
    letters_need_unk = keys_for_unk.difference(keys_for_comparing)
    letters_need_comparing = keys_for_comparing.difference(keys_for_unk)
    for i in letters_need_unk:
        comparing_text.setdefault(i, 0.0)
    for x in letters_need_comparing:
        unknown_text.setdefault(x, 0.0)
    sorted_unk = dict(sorted(unknown_text.items()))
    sorted_comparing = {}
    for key in sorted_unk:
        sorted_comparing[key] = comparing_text.get(key, unknown_text[key])
    list_unk = list(sorted_unk.values())
    list_comparing = list(sorted_comparing.values())
    mse = calculate_mse(list_comparing, list_unk)
    if not isinstance(mse, float):
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
    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    if (unknown_profile is None and profile_1 is None
            and profile_2 is None):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if (not isinstance(mse_1, float)
            or not isinstance(mse_2,  float)):
        return None
    if mse_1 < mse_2:
        return str(profile_1["name"])
    if mse_1 > mse_2:
        return str(profile_2["name"])
    return str(profile_1["name"])


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
    with open(path_to_file, 'r', encoding='utf-8') as file_to_read:
        dictionary = json.load(file_to_read)
        if not isinstance(dictionary, dict):
            return None
        return dictionary


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
    if not isinstance(profile, dict):
        return None
    if (len(profile.keys()) != 3
            or not all(x in profile for x in ['freq', 'name', 'n_words'])):
        return None
    processed_profile = {'name': profile['name'], 'freq': {}}
    dictionary: dict[str, int] = {}
    for unigram in profile['freq'].keys():
        if len(unigram) == 1:
            dictionary.setdefault(unigram, profile['freq'][unigram])
    letters = list(dictionary)
    for letter in letters:
        if not letter.isupper():
            dictionary[letter] = dictionary.get(letter, 0) + dictionary.get(letter.upper(), 0)
            processed_profile['freq'][letter.lower()] = dictionary[letter] / profile['n_words'][0]
            if letter.upper() in dictionary:
                del dictionary[letter.upper()]
    for i in dictionary.items():
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
    if (not isinstance(paths_to_profiles, list)
            or not all(isinstance(letter, str) for letter in paths_to_profiles)):
        return None
    list_for_dictionaries = []
    for name_of_profile in paths_to_profiles:
        if not load_profile(name_of_profile):
            return None
        dictionary_unprocessed = load_profile(name_of_profile)
        if not dictionary_unprocessed:
            return None
        new_load: dict = dictionary_unprocessed
        processed_profile = preprocess_profile(new_load)
        if not isinstance(processed_profile, dict):
            return None
        list_for_dictionaries.append(processed_profile)
    return list_for_dictionaries


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
    if not isinstance(unknown_profile, dict):
        return None
    if (len(unknown_profile.keys()) != 2
            or not all(k in unknown_profile for k in ('freq', 'name'))):
        return None
    if (not isinstance(known_profiles, list) and
            not all(isinstance(path, dict) for path in known_profiles)):
        return None
    sorted_list = []
    for profile in known_profiles:
        sorted_list.append((profile["name"],
                            compare_profiles(unknown_profile, profile)))
    if sorted_list:
        return sorted(sorted_list, key=lambda block: (block[1], block[0]))
    return None


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    for block in detections:
        print(f'{block[0]}: MSE {block[1]:.5f}')
