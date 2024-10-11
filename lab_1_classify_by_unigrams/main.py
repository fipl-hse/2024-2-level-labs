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
    for symb in text:
        if symb.isalpha():
            tokens.append(symb.lower())
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

    if not isinstance(tokens, list) or len(tokens) == 0 or None in tokens:
        return None
    freq = {}
    all_tokens = len(tokens)
    token_freq = 1 / all_tokens
    for token in tokens:
        if token not in freq:
            freq[token] = 0.0
        freq[token] += token_freq
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

    if not (isinstance(language, str) and isinstance(text, str)):
        return None
    if tokenize(text) is None:
        return None
    freq_dict = calculate_frequencies(tokenize(text))
    if not isinstance(freq_dict, dict):
        return None
    return {'name': language, 'freq': freq_dict}


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
    summ = 0
    n = len(actual)
    for i in range(0, n):
        squared_difference = (actual[i] - predicted[i]) ** 2
        summ = summ + squared_difference
    return summ / n


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

    if not unknown_profile or not profile_to_compare:
        return None
    if (not isinstance(unknown_profile['freq'], dict)
            and not isinstance(profile_to_compare['freq'], dict)):
        return None
    if not ('name' in unknown_profile and 'name' in profile_to_compare):
        return None
    if not ('freq' in unknown_profile and 'freq' in profile_to_compare):
        return None
    all_tokens = set(unknown_profile['freq'].keys()).union(set(profile_to_compare['freq'].keys()))
    freq_first_lang = [unknown_profile['freq'].get(token, 0) for token in all_tokens]
    freq_second_lang = [profile_to_compare['freq'].get(token, 0) for token in all_tokens]
    return calculate_mse(freq_first_lang, freq_second_lang)


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

    if (not isinstance(unknown_profile, dict) and not isinstance(profile_1, dict)
            and not isinstance(profile_2, dict)):
        return None
    if (unknown_profile is None or profile_1 is None
            or profile_2 is None):
        return None
    if (isinstance(unknown_profile, dict) and isinstance(profile_1, dict)
            and isinstance(profile_2, dict)):
        mse_prof_1 = compare_profiles(unknown_profile, profile_1)
        mse_prof_2 = compare_profiles(unknown_profile, profile_2)
        if mse_prof_1 is not None and mse_prof_2 is not None:
            prof_1_name = profile_1.get('name')
            prof_2_name = profile_2.get('name')
            if isinstance(prof_1_name, str) and isinstance(prof_2_name, str):
                if mse_prof_1 < mse_prof_2:
                    return prof_1_name
                if mse_prof_1 > mse_prof_2:
                    return prof_2_name
                return min(prof_1_name, prof_2_name)
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
