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

    text = text.lower()
    symb = ',./?!:;#@*-&<>% '
    tokens = []
    for elem in text:
        if not elem.isdigit() and elem not in symb:
            tokens.append(elem)
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
    token_freq = {}
    if not isinstance(tokens, list):
        return None

    if not tokens:
        return None

    for letter in tokens:
        if not isinstance(letter, str) or len(letter) > 1:
            return None
        if letter not in token_freq:
            token_freq[letter] = 1
        else:
            token_freq[letter] += 1

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
    if not tokenize(text):
        return None
    if not isinstance(language, str):
        return None
    return {'name': language, 'freq': calculate_frequencies(tokenize(text))}


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
    mse_sum = 0

    if len(predicted) != len(actual):
        return None

    for i, a in enumerate(predicted):
        mse_sum += (a - actual[i]) ** 2

    return round(mse_sum / len(predicted), 4)


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
    if ("name" not in unknown_profile
            or "freq" not in unknown_profile
            or "name" not in profile_to_compare
            or "freq" not in profile_to_compare):
        return None
    if (not isinstance(profile_to_compare["name"], str)
            or not isinstance(profile_to_compare["freq"], dict)):
        return None
    if (not isinstance(unknown_profile["name"], str)
            or not isinstance(unknown_profile["freq"], dict)):
        return None

    unknown_freq = unknown_profile['freq']
    compare_freq = profile_to_compare['freq']
    all_tokens = set(unknown_freq).union(set(compare_freq))

    unknown_shared = []
    for token in all_tokens:
        if token in unknown_freq:
            unknown_shared.append(unknown_freq[token])
        else:
            unknown_shared.append(0.0)

    compare_shared = []
    for token in all_tokens:
        if token in compare_freq:
            compare_shared.append(compare_freq[token])
        else:
            compare_shared.append(0.0)

    return calculate_mse(unknown_shared, compare_shared)


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
    if not (isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if isinstance(mse_1, float) and isinstance(mse_2, float):
        if mse_1 > mse_2 and isinstance(profile_2['name'], str):
            return profile_2['name']
        if mse_2 > mse_1 and isinstance(profile_1['name'], str):
            return profile_1['name']
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
