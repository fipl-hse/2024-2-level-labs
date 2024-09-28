"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from json import load
from string import punctuation


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

    # checking input
    if not isinstance(text, str):
        return None

    text = ''.join(text.split()).lower()

    for char in punctuation + '1234567890º’':
        text = text.replace(char, '')

    return list(text)


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """

    # checking input
    if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens) \
            or not tokens:
        return None

    tokens_total = len(tokens)

    freq = {}
    for char in set(tokens):
        freq[char] = float(tokens.count(char)) / tokens_total

    return freq


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

    # checking input
    if not isinstance(language, str) or not isinstance(text, str):
        return None

    frequencies = calculate_frequencies(tokenize(text))
    if (not isinstance(frequencies, dict)
            or not all(isinstance(char, str) for char in frequencies)
            or not all(isinstance(value, float) for value in frequencies.values())):
        return None

    if frequencies is not None:
        return {'name': language, 'freq': frequencies}
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

    # checking input
    if not isinstance(predicted, list) or not isinstance(actual, list) \
            or len(predicted) != len(actual):
        return None

    return float(sum((actual[i] - predicted[i]) ** 2
                     for i in range(len(actual))) / len(actual))


def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]], ) -> float | None:
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

    # checking input
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if 'name' not in unknown_profile or 'name' not in profile_to_compare:
        return None
    if not all(isinstance(key, str) for key in unknown_profile) \
            or not all(isinstance(key, str) for key in profile_to_compare):
        return None

    all_chars = set(unknown_profile['freq']).union(set(profile_to_compare['freq']))
    # creating lists to pass into calculate_mse function
    unknown_profile_freq = unknown_profile['freq']
    if not isinstance(unknown_profile_freq, dict):
        return None
    unknown_sorted = [(unknown_profile_freq[char] if char in unknown_profile_freq else 0.0)
                      for char in all_chars]

    profile_to_compare_freq = profile_to_compare['freq']
    if not isinstance(profile_to_compare_freq, dict):
        return None
    profile_to_compare_sorted = [(profile_to_compare_freq[char]
                                  if char in profile_to_compare_freq else 0.0)
                                 for char in all_chars]

    return calculate_mse(unknown_sorted, profile_to_compare_sorted)


def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]]) -> str | None:
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

    # checking input
    if not all(isinstance(profile, dict) for profile in [unknown_profile, profile_1, profile_2]) \
            or not all(isinstance(key, str) for key in unknown_profile) \
            or not all(isinstance(key, str) for key in profile_1) \
            or not all(isinstance(key, str) for key in profile_2):
        return None

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if not isinstance(mse_1, float) or not isinstance(mse_2, float):
        return None

    # if mse values differ
    if mse_1 > mse_2:
        return str(profile_2['name'])
    if mse_2 > mse_1:
        return str(profile_1['name'])
    # if mse values are the same -> return the first one by alphabetical order
    if str(profile_1['name']) > str(profile_2['name']):
        return str(profile_2['name'])
    return str(profile_1['name'])


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """

    # checking input
    if not isinstance(path_to_file, str):
        return None

    with open(path_to_file, 'r', encoding='UTF-8') as file:
        profile = load(file)

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

    # checking input
    if not isinstance(profile, dict) or not \
            all(key in profile for key in ['name', 'n_words', 'freq']):
        return None

    profile_preprocessed = {'name': profile['name'], 'freq': {}}

    all_unigrams_count = profile['n_words'][0]

    for key in profile['freq']:
        if len(key.strip()) == 1:
            if key.lower() in profile_preprocessed['freq']:
                profile_preprocessed['freq'][key.lower()] += profile['freq'][key] \
                                                             / all_unigrams_count
            else:
                profile_preprocessed['freq'][key.lower()] = profile['freq'][key] \
                                                            / all_unigrams_count

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

    # checking input
    if not isinstance(paths_to_profiles, list):
        return None

    loaded_profiles = []
    for path in paths_to_profiles:
        profile_loaded = load_profile(path)
        if isinstance(profile_loaded, dict):
            profile_preprocessed = preprocess_profile(profile_loaded)
            if isinstance(profile_preprocessed, dict):
                loaded_profiles.append(profile_preprocessed)
        else:
            return None

    return loaded_profiles


def detect_language_advanced(
        unknown_profile: dict[str, str | dict[str, float]], known_profiles: list) -> list | None:
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

    # checking input
    if not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list):
        return None

    distances = []
    for profile in known_profiles:
        profile_name: str = profile['name']
        distance = compare_profiles(unknown_profile, profile)
        if distance is None:
            return None
        distances.append((profile_name, distance))
    distances_sorted: list[tuple['str', float]] = sorted(distances, key=lambda t: t[1])
    return distances_sorted


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """

    # checking input
    if not isinstance(detections, list):
        return None

    for result in detections:
        print(f'{result[0]}: MSE {result[1]:.5f}')

    return None
