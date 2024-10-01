"""
Lab 1.
Language detection
"""

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
    return [symbol.lower() for symbol in text if symbol.isalpha()]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
       Calculate frequencies of given tokens.
       Args:
           tokens (list[str] | None): A list of tokens
       Returns:
           dict[str, float] | None: A dictionary with frequencies
       In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or any(not isinstance(token, str) for token in tokens):
        return None
    frequency = {}
    for token in tokens:
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
    tokens = tokenize(text)
    frequency = calculate_frequencies(tokens)
    if tokens is None or not isinstance(text, str) or any(not isinstance(
            token, str) for token in tokens) or frequency is None:
        return None
    return {'name': language, 'freq': frequency}


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
    if len(actual) == 0 or len(actual) != len(predicted):
        return None
    if not all(isinstance(predicted_values, (
            int, float)) for predicted_values in predicted) or not all(
        isinstance(actual_values, (int, float)) for actual_values in actual):
        return None
    return float(sum((actual[i] - predicted[i]) ** 2 for i in range(
        len(actual))) / len(actual))


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
    if 'name' not in unknown_profile or 'freq' not in unknown_profile:
        return None
    if 'name' not in profile_to_compare or 'freq' not in profile_to_compare:
        return None
    unknown_freq = unknown_profile['freq']
    compare_freq = profile_to_compare['freq']
    if not isinstance(unknown_freq, dict) or not isinstance(compare_freq, dict):
        return None
    tokens = set(unknown_freq.keys()).union(set(compare_freq.keys()))
    actual_values = []
    predicted_values = []
    for token in tokens:
        actual_values.append(unknown_freq.get(token, 0))
        predicted_values.append(compare_freq.get(token, 0))
    return calculate_mse(predicted_values, actual_values)


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
    if not all(isinstance(profile, dict) and 'freq' in profile and
               'name' in profile for profile in (unknown_profile, profile_1, profile_2)):
        return None
    if not all(isinstance(profile['name'], str) for profile in (profile_1, profile_2)):
        return None
    distance_1 = compare_profiles(unknown_profile, profile_1)
    distance_2 = compare_profiles(unknown_profile, profile_2)
    if distance_1 is None:
        return (str(profile_2['name'])) if distance_2 is not None else None
    if distance_2 is None or distance_1 < distance_2:
        return str(profile_1['name'])
    if profile_2 is None or profile_1 is None:
        return None
    return str(profile_2['name']) if distance_2 < distance_1 else min(
        str(profile_1['name']), str(profile_2['name']))


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
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
    return None


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    return None


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
    return None


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    return None
