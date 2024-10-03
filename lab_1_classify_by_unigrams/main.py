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
    lowered_text = text.lower()
    tokens = []
    for elem in lowered_text:
        if elem.isalpha() and elem != "º":
            tokens += elem
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
    if isinstance(tokens, list) is False:
        return None

    for elem in tokens:
        if isinstance(elem, str) is False:
            return None

    token_dict = {}
    for i in tokens:
        for i in token_dict:
            if i not in token_dict:
                token_dict.update({i: 0.0})
                token_dict[i] += 1.0

    sum_token_dict_values = sum(token_dict.values())
    for i in token_dict.items():
        new_elem = token_dict[i] / sum_token_dict_values
        token_dict[i] = new_elem
    return token_dict


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
    if not ((isinstance(language, str)) and (isinstance(text, str))):
        return None

    if calculate_frequencies(tokenize(text)) is None:
        return None

    language_prof = {"name": language,
                     "freq": calculate_frequencies(tokenize(text))
                     }
    return language_prof


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
    if not (isinstance(predicted, list) or isinstance(actual, list)):
        return None
    if len(predicted) != len(actual):
        return None

    mse = 0
    for i in enumerate(predicted):
        mse += (predicted[i] - actual[i])**2

    return mse/len(predicted)


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
    if ((isinstance(unknown_profile, dict) is False)
            or (isinstance(profile_to_compare, dict) is False)):
        return None
    if ("name" not in unknown_profile) or ("name" not in profile_to_compare):
        return None
    for char_percentage in unknown_profile["freq"]:
        if char_percentage not in profile_to_compare["freq"]:
            return profile_to_compare["freq"].update({char_percentage: 0})
    for char_percentage in profile_to_compare["freq"]:
        if char_percentage not in unknown_profile["freq"]:
            return unknown_profile["freq"].update({char_percentage: 0})

    if len(unknown_profile["freq"]) == len(profile_to_compare["freq"]):
        unknown_profile_list = list(dict(sorted(unknown_profile["freq"].items())).values())
        profile_to_compare_list = list(dict(sorted(profile_to_compare["freq"].items())).values())
        mse = calculate_mse(unknown_profile_list, profile_to_compare_list)
        return mse
    return mse


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
            and isinstance(profile_1, dict) and isinstance(profile_2, dict)):
        return None

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 < mse_2:
        unknown_profile["name"] = profile_1["name"]
    else:
        unknown_profile["name"] = profile_2["name"]
    return unknown_profile["name"]


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
