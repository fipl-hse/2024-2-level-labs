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
    tokenized_text = []
    for symbol in text:
        if symbol.isalpha():
            tokenized_text.append(symbol)
    return tokenized_text



def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or None in tokens:
        return None
    frequency_dictionary = {}
    for token in tokens:
        if token not in frequency_dictionary:
            frequency_dictionary[token] = tokens.count(token)/len(tokens)
    return frequency_dictionary

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
    if not isinstance(text, str) or not isinstance(language, str):
        return None
    frequencies = calculate_frequencies(tokenize(text))
    if not isinstance(frequencies, dict):
        return None
    return {'name': language,
            'freq': frequencies}

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
    if not isinstance(predicted, list) or not isinstance(actual, list) or \
            not len(predicted) == len(actual):
        return None
    diff = 0
    for m, n in enumerate(actual):
        diff += (n - predicted[m])**2
    return diff / len(actual)

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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict) or \
            len(profile_to_compare) != len(unknown_profile) != 2:
        return None
    unknown_lst = []
    compared_lst = []
    if not isinstance(unknown_profile['freq'], dict) \
            or not isinstance(profile_to_compare['freq'], dict):
        return None
    for letter in unknown_profile['freq']:
        if letter not in profile_to_compare['freq']:
            profile_to_compare['freq'][letter] = 0
    for letter in profile_to_compare['freq']:
        if letter not in unknown_profile['freq']:
            unknown_profile['freq'][letter] = 0
    unknown_sorted = dict(sorted(unknown_profile['freq'].items()))
    compared_sorted = dict(sorted(profile_to_compare['freq'].items()))
    if not isinstance(unknown_profile, dict) or not isinstance(compared_sorted, dict):
        return None
    for i in unknown_sorted.values():
        unknown_lst.append(i)
    for i in compared_sorted.values():
        compared_lst.append(i)
    return calculate_mse(compared_lst, unknown_lst)




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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict)\
            or not isinstance(profile_2, dict):
        return None
    mse_unk_p1 = compare_profiles(unknown_profile, profile_1)
    mse_unk_p2 = compare_profiles(unknown_profile, profile_2)
    if not isinstance(mse_unk_p1, float) or not isinstance(mse_unk_p2, float):
        return None
    if not isinstance(profile_1['name'], str) or not isinstance(profile_2['name'], str):
        return None
    if mse_unk_p1 > mse_unk_p2:
        return profile_2.get('name')
    if mse_unk_p1 < mse_unk_p2:
        return profile_1.get('name')
    if mse_unk_p1 == mse_unk_p2:
        sort_lang = [profile_1['name'], profile_2['name']]
        sort_lang.sort()
        return sort_lang[0]
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
