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
    token_list = []
    for symbol in text:
        if symbol.isalpha() and symbol != "º":
            token_list.append(symbol)
    return token_list


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

    freq_dic = {}

    for check in tokens:
        if not isinstance(check, str):
            return None
        freq_dic[check] = tokens.count(check) / len(tokens)

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

    if (not isinstance(text, str)
            or not isinstance(language, str)):
        return None

    freq = calculate_frequencies(tokenize(text))

    if not isinstance(freq, dict):
        return None

    return {"name": language, "freq": freq}


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

    if (not isinstance(predicted, list)
            or not isinstance(actual, list)):
        return None

    total_number = len(actual)

    if len(predicted) != total_number:
        return None

    result = 0

    for index in range(total_number):
        step = (predicted[index] - actual[index]) ** 2
        result += step
    return result / total_number


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

    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_to_compare, dict)):
        return None
    if ("name" not in unknown_profile
            or "freq" not in unknown_profile
            or "name" not in profile_to_compare
            or "freq" not in profile_to_compare):
        return None
    if (not isinstance(profile_to_compare["name"], str)
            or not isinstance(profile_to_compare["freq"], dict)):
        return None

    unknown_freq = unknown_profile["freq"]
    compare_freq = profile_to_compare["freq"]
    unknown_keys = list(unknown_freq.keys())
    compare_keys = list(compare_freq.keys())
    all_keys = (unknown_keys + list(set(compare_keys) - set(unknown_keys)))
    all_unknown_profile = dict.fromkeys(all_keys, 0.0)
    all_profile_to_compare = dict.fromkeys(all_keys, 0.0)
    for (key, value) in unknown_freq.items():
        for (all_key, all_value) in all_unknown_profile.items():
            if key == all_key:
                all_unknown_profile[all_key] = float(value)
    for (key, value) in compare_freq.items():
        for (all_key, all_value) in all_profile_to_compare.items():
            if key == all_key:
                all_profile_to_compare[all_key] = float(value)
    return calculate_mse(list(all_unknown_profile.values()),
                         list(all_profile_to_compare.values()))


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

    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None

    mse_unknown_to_1 = compare_profiles(unknown_profile, profile_1)
    mse_unknown_to_2 = compare_profiles(unknown_profile, profile_2)
    result = ''

    if mse_unknown_to_1 is None or mse_unknown_to_2 is None:
        return None

    if mse_unknown_to_1 < mse_unknown_to_2:
        if isinstance(profile_1["name"], str):
            result = profile_1["name"]
    if mse_unknown_to_2 < mse_unknown_to_1:
        if isinstance(profile_2["name"], str):
            result = profile_2["name"]
    if mse_unknown_to_1 == mse_unknown_to_2:
        asort = [profile_1["name"], profile_2["name"]]
        asort.sort()
        if isinstance(asort[0], str):
            result = asort[0]
    return result


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
