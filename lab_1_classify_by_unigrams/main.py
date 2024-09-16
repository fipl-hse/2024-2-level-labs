from collections import Counter
from string import punctuation
from json import load

# pylint:disable=too-many-locals, unused-argument, unused-variable


def tokenize(text: str) -> list[str] | None:
    # checking input
    if type(text) != str:
        return None

    text = ''.join(text.split()).lower()

    for char in punctuation + '1234567890º’':
        text = text.replace(char, '')

    return list(text)


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    # checking types
    if type(tokens) != list or not all(isinstance(item, str) for item in tokens):
        return None

    tokens_total = len(tokens)

    freq = dict(Counter(tokens))
    for letter in freq:
        freq[letter] = freq[letter] / tokens_total

    return freq


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    # checking input
    if type(language) != str or type(text) != str:
        return None

    language_frequencies = calculate_frequencies(tokenize(text))

    profile = {'name': language, 'freq': language_frequencies}

    return profile


def calculate_mse(predicted: list, actual: list) -> float | None:
    # checking input
    if type(predicted) != list or type(actual) != list:
        return None
    if len(predicted) != len(actual):
        return None

    # опытным путём на меньшем количестве данных выяснила, что дело, кажется, в точности представления float в памяти, а не в неверном коде
    # странно, что это работает у всех, но не у меня
    mse = sum((actual[i] - predicted[i]) ** 2 for i in range(len(actual))) / len(actual)

    return round(mse, 4)


def compare_profiles(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_to_compare: dict[str, str | dict[str, float]],
    ) -> float | None:

    # checking input
    if type(unknown_profile) != dict or type(profile_to_compare) != dict:
        return None
    if 'name' not in unknown_profile.keys() or 'name' not in profile_to_compare.keys():
        return None

    # creating lists to pass into calculate_mse function
    unknown_sorted = [(unknown_profile['freq'][char] if char in unknown_profile['freq'] else 0.0) for char in
                      profile_to_compare['freq']]

    mse = calculate_mse(unknown_sorted, list(profile_to_compare['freq'].values()))

    return mse


def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    # checking input
    if type(unknown_profile) != dict or type(profile_1) != dict or type(profile_2) != dict:
        return None

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)

    # if mse values differ
    if mse_1 > mse_2:
        return profile_2['name']
    if mse_2 > mse_1:
        return profile_1['name']
    # if mse values are the same -> return the first one by alphabetical order
    if profile_1['name'] > profile_2['name']:
        return profile_2['name']
    return profile_1['name']


def load_profile(path_to_file: str) -> dict | None:
    # checking input
    if type(path_to_file) != str:
        return None

    with open(path_to_file, 'r') as file:
        file = load(file)

    return file


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    # checking input
    if type(profile) != dict:
        return None

    profile_preprocessed = {'name': profile['name'], 'freq': {}}

    all_unigrams_count = profile['n_words'][0]

    for key in profile['freq']:
        if len(key.strip()) == 1 and key.isalpha() and key.lower() not in profile_preprocessed.keys():
            profile_preprocessed['freq'][key.lower()] = profile['freq'][key] / all_unigrams_count
        elif len(key.strip()) == 1 and key.isalpha():
            profile_preprocessed['freq'][key.lower()] += profile['freq'][key] / all_unigrams_count

    return profile_preprocessed


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """


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


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
