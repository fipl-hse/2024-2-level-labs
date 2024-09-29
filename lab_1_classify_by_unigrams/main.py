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
    ab = 'qwertyuioplkjhgfdsazxcvbnmMNBVCXZASDFGHJKLPOIUYTREWQÄäẞßÜüÖö'
    out = []
    for char in text:
        if char in ab:
            out.append(char.lower())
    if not out:
        return None
    return out


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
    frq = {}
    for sy in tokens:
        if not isinstance(sy, str):
            return None
        frq.update({sy: tokens.count(sy) / len(tokens)})
    return frq


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
    if len(text) <= 0:
        return None
    out = {'name': language, 'freq': calculate_frequencies(tokenize(text))}
    return out


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
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    sig = 0
    for s, (src, trg) in enumerate(zip(predicted, actual)):
        sig += ((trg - src) ** 2)
    if len(actual) == 0:
        return None
    if len(actual) != len(predicted):
        return None
    return sig / len(actual)


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
    if (list(unknown_profile.keys())[0] != 'name') or (list(unknown_profile.keys())[1] != 'freq'):
        return None
    if list(profile_to_compare.keys())[0] != 'name':
        return None
    if list(profile_to_compare.keys())[1] != 'freq':
        return None
    trg_prof = list(unknown_profile.values())[1]
    src_prof = list(profile_to_compare.values())[1]
    if not isinstance(trg_prof, dict) or not isinstance(src_prof, dict):
        return None
    for tv in list(trg_prof.keys()):
        if src_prof.get(tv, 0.0) == 0.0:
            src_prof.update({tv: 0.0})
    for tv in list(src_prof.keys()):
        if trg_prof.get(tv, 0.0) == 0.0:
            trg_prof.update({tv: 0.0})
    trg_srt = dict(sorted(trg_prof.items()))
    src_srt = dict(sorted(src_prof.items()))
    out = calculate_mse(list(src_srt.values()), list(trg_srt.values()))
    return out


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
    if not isinstance(unknown_profile, dict):
        return None
    if not isinstance(profile_1, dict):
        return None
    if not isinstance(profile_2, dict):
        return None
    check1 = compare_profiles(unknown_profile, profile_1)
    check2 = compare_profiles(unknown_profile, profile_2)
    if check1 < check2:
        return list(profile_1.values())[0]
    if check2 < check1:
        return list(profile_2.values())[0]
    failsafe = [list(profile_1.values())[0], list(profile_2.values())[0]]
    failsafe.sort(key=str.lower)
    return failsafe[0]


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


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
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
