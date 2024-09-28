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
    tokens = []
    for token in text.lower():
        if token.isalpha():
            tokens.append(token)
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
    if (not isinstance(tokens, list)
            or next((token for token in tokens if isinstance(token, str)), None) is None):
        return None
    freq_list = []
    token_list = []
    freq_dic = {}
    for token in tokens:
        freq_list.append(tokens.count(token) / len(tokens))
        token_list.append(token)
        freq_dic = dict(zip(token_list, freq_list))
    return freq_dic


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
    if (not isinstance(language, str) or not isinstance(text, str)
            or not isinstance(tokenize(text), list)):
        return None
    calced_freq = calculate_frequencies(tokenize(text))

    if not isinstance(calced_freq, dict):
        return None
    return dict(zip(['name', 'freq'], [language, calced_freq]))


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
            or len(predicted) != len(actual)):
        return None
    squared_error_sum = sum((p - a) ** 2 for p, a in zip(predicted, actual))
    if not isinstance(squared_error_sum, float):
        return None
    return squared_error_sum / len(predicted)


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
    if (not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict)
            or not all(key in unknown_profile for key in ('name', 'freq'))
            or not all(key in profile_to_compare for key in ('name', 'freq'))):
        return None
    tokens_union = set(profile_to_compare['freq'].keys()).union(unknown_profile['freq'].keys())
    unknown_freq = [unknown_profile['freq'].get(token, 0) for token in tokens_union]
    freq_to_compare = [profile_to_compare['freq'].get(token, 0) for token in tokens_union]
    mse = calculate_mse(freq_to_compare, unknown_freq)
    if not isinstance(mse, float):
        return None
    return round(mse, 3)


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
    if (any(not isinstance(profile, dict) for profile in (unknown_profile, profile_1, profile_2))
        or not all(key in unknown_profile for key in ('name', 'freq'))
            or not all(key in profile_1 for key in ('name', 'freq'))
            or not all(key in profile_2 for key in ('name', 'freq'))):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if (not isinstance(profile_1['name'], str) or not isinstance(profile_2['name'], str)
            or any(not isinstance(mse, float) for mse in (mse_1, mse_2))
            or mse_1 is None or mse_2 is None):
        return None
    if mse_1 < mse_2:
        return profile_1['name']
    if mse_1 > mse_2:
        return profile_2['name']
    if mse_1 == mse_2:
        profiles_list = sorted([profile_1['name'], profile_2['name']])
        return profiles_list[0]
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
