"""
Lab 1.

Language detection
"""
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
    if not isinstance(text, str):
        return None
    needless = "\n1234567890,.:; -!?'*º’‘@~#№$%^&<>|+-"
    text = text.lower()
    tokens = []
    for i in text:
        if i not in needless:
            tokens += i
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
    if not isinstance(tokens, list):
        return None
    freq = {}
    for i in tokens:
        if not isinstance(i, str):
            return None
        if len(i) != 1:
            return None
        if i in freq:
            continue
        freq[i] = tokens.count(i) / len(tokens)
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
    tokens = tokenize(text)
    freq = calculate_frequencies(tokens)
    if freq is None or tokens is None or not isinstance(language, str):
        return None
    return {'name': language, 'freq': freq}


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
    if len(actual) != len(predicted):
        return None
    squa_diff = 0
    for i, x in enumerate(actual):
        squa_diff += (x - predicted[i]) ** 2
    return round((squa_diff/len(actual)), 4)


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
        return  None
    if 'name' not in unknown_profile or 'name' not in profile_to_compare:
        return  None
    for elem in unknown_profile['freq'].keys():
        if elem not in profile_to_compare['freq']:
            if not isinstance(profile_to_compare['freq'], dict):
                return None
            profile_to_compare['freq'][elem] = 0.0
    for elem in profile_to_compare['freq'].keys():
        if elem not in unknown_profile['freq']:
            if not isinstance(unknown_profile['freq'], dict):
                return None
            unknown_profile['freq'][elem] = 0.0
    profile_to_compare['freq'] = dict(sorted(profile_to_compare['freq'].items()))
    unknown_profile['freq'] = dict(sorted(unknown_profile['freq'].items()))
    predicted = []
    for elem in unknown_profile['freq'].values():
        predicted.append(elem)
    actual = []
    for elem in profile_to_compare['freq'].values():
        actual.append(elem)
    profile_mse = calculate_mse(predicted, actual)
    return profile_mse


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
    detect_1 = compare_profiles(unknown_profile, profile_1)
    detect_2 = compare_profiles(unknown_profile, profile_2)
    if not isinstance(detect_1, float):
        return None
    if not isinstance(detect_2, float):
        return None
    if detect_1 < detect_2:
        lang = str(profile_1['name'])
    elif detect_1 == detect_2:
        a = [profile_1['name'], profile_2['name']]
        a.sort()
        lang = str(a[0])
    else:
        lang = str(profile_2['name'])
    return lang


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """


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
