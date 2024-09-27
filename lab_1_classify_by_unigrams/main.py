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

    tokens_list = [symbol.lower() for symbol in text if symbol.isalpha()]
    return tokens_list


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not (isinstance(tokens, list) and all(isinstance(x, str) for x in tokens)):
        return None

    freq_dict = {}
    for i in tokens:
        freq_dict.setdefault(i, 0)
        freq_dict[i] += 1
    for n in freq_dict:
        freq_dict[n] /= len(tokens)

    return freq_dict


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

    freq_dict = calculate_frequencies(tokenize(text))
    if freq_dict is None:
        return None

    return {"name": language, "freq": freq_dict}


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
    if not (isinstance(predicted, list) and
            isinstance(actual, list) and
            len(predicted) == len(actual) and
            len(predicted) == 0):
        return None

    num_of_diffs = len(predicted)
    sum_of_diffs = 0
    for i in range(num_of_diffs):
        sum_of_diffs += (actual[i] - predicted[i]) ** 2
    mse = sum_of_diffs / num_of_diffs

    return mse


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
    if not (isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict) and
            all("name" in a for a in (unknown_profile, profile_to_compare)) and
            all("freq" in b for b in (unknown_profile, profile_to_compare))):
        return None
    
    all_keys = list(set(unknown_profile["freq"]) | set(profile_to_compare["freq"]))
    unknown_profile_with_0s = []
    profile_to_compare_with_0s = []
    for i in all_keys:
        if not i in unknown_profile["freq"]:
            unknown_profile_with_0s.append(0)
        else:
            unknown_profile_with_0s.append(unknown_profile["freq"][i])
        if not i in profile_to_compare["freq"]:
            profile_to_compare_with_0s.append(0)
        else:
            profile_to_compare_with_0s.append(profile_to_compare["freq"][i])

    return calculate_mse(unknown_profile_with_0s, profile_to_compare_with_0s)


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
    if not (isinstance(unknown_profile, dict) and
            isinstance(profile_1, dict) and
            isinstance(profile_2, dict)):
        return None

    diff_unk_1 = compare_profiles(unknown_profile, profile_1)
    diff_unk_2 = compare_profiles(unknown_profile, profile_2)
    if diff_unk_1 is None or diff_unk_2 is None:
        return None

    if diff_unk_1 < diff_unk_2:
        res = str(profile_1["name"])
        return res
    res = str(profile_2["name"])
    return res


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
