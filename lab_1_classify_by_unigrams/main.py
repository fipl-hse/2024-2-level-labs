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

    if text is None or type(text) is not str:
        return None

    tozenized_text = []
    bad_symbols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.', '-', '/', ':', ';', '<', '=', '>',
                        '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']

    for symb in text:
        if symb in bad_symbols:
            continue
        low_symb = symb.lower()
        tozenized_text.append(low_symb)

    if not tozenized_text:
        return None
    else:
        return tozenized_text

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """

    if tokens is None or not isinstance(tokens, list):
        return  None
    frequencies = {}
    for token in tokens:
        if not isinstance(token, str):
            return None
        if token not in frequencies:
            frequencies[token] = 0
        frequencies[token] += 1

    for token in frequencies:
        frequencies[token] = frequencies[token] / len(tokens)

    return frequencies



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
    profile = {}
    if not isinstance(language, str) or language not in ['en', 'de'] or not isinstance(text, str):
        return None
    profile['name'] = language
    profile['freq'] = calculate_frequencies(tokenize(text))
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

    if not isinstance(predicted, list) or not isinstance(actual, list) or len(predicted) != len(actual):
        return None
    n = len(predicted)
    mse = 0
    for i in range(n):
        mse += (predicted[i] - actual[i]) ** 2
    return mse / n

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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict) or unknown_profile.get('name') is None or profile_to_compare.get('name') is None or profile_to_compare.get('freq') is None or unknown_profile.get('freq') is None:
        return None

    lang_1 = unknown_profile['freq']
    lang_2 = profile_to_compare['freq']

    for element in lang_1:
        if element not in lang_2:
            lang_2[element] = 0

    for element in lang_2:
        if element not in lang_1:
            lang_1[element] = 0

    sorted_lang_1 = []
    sorted_lang_2 = []

    sorted_lang_1 = list(lang_1.values())

    for letter in lang_1:
        sorted_lang_2.append(lang_2[letter])

    return calculate_mse(sorted_lang_1, sorted_lang_2)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) or not isinstance(profile_2, dict):
        return None

    score_1 = compare_profiles(unknown_profile, profile_1)
    score_2 = compare_profiles(unknown_profile, profile_2)

    if score_1 < score_2:
        return profile_1['name']
    elif score_1 == score_2:
        print('de')
    else:
        return profile_2['name']



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

