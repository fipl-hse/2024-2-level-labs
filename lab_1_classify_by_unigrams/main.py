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
        return
    else:
        text = text.lower().split()
        split2 = []
        for word in text:
            for token in word:
                if token.isalpha():
                    split2 += token
        return split2


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
    else:
        for token in tokens:
            if not isinstance(token, str):
                return None
        freq = {}
        prop = {}
        total_tokens = len(tokens)
        for token in tokens:
            if token in freq:
                freq[token] += 1
            else:
                freq[token] = 1
        for key, value in freq.items():
            prop[key] = value / total_tokens
        return prop


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
    return {
        'name': language,
        'freq': calculate_frequencies(tokenize(text))
    }


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
    if not isinstance(predicted, list) or not isinstance(actual,list) or (len(predicted) != len(actual)):
        return None
    sum_diff = 0
    for pair_num, actual_num in enumerate(actual):
        sum_diff += (actual_num-predicted[pair_num])**2
    return sum_diff / len(predicted)


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
    if not (isinstance(unknown_profile, dict)) or not (isinstance(profile_to_compare, dict)) or not (len(unknown_profile) ==2) or not (len(profile_to_compare) == 2):
        return None
    for i in unknown_profile['freq']:
        if i not in profile_to_compare['freq']:
            profile_to_compare['freq'][i] = 0
    for i in profile_to_compare['freq']:
        if i not in unknown_profile['freq']:
            unknown_profile['freq'][i] = 0
    sorted_unknown = dict(sorted(unknown_profile['freq'].items()))
    sorted_compare = dict(sorted(profile_to_compare['freq'].items()))
    sort_comp_list = []
    sort_unkn_list = []
    for i in sorted_unknown.values():
        sort_unkn_list.append(i)
    for i in sorted_compare.values():
        sort_comp_list.append(i)
    return calculate_mse(sort_unkn_list, sort_comp_list)


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
    if not (isinstance(unknown_profile, dict)) or not (isinstance(profile_1, dict)) or not (isinstance(profile_2,dict)) or not (
                len(unknown_profile) == 2) or not (len(profile_1) == 2)  or not (len(profile_2) == 2):
        return None
    else:
        mse_1 = compare_profiles(unknown_profile, profile_1)
        mse_2 = compare_profiles(unknown_profile, profile_2)
        if mse_1 < mse_2:
            return (profile_1["name"])
        else:
            return (profile_2["name"])


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
