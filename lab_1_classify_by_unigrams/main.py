"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

def tokenize(text):
    token_list = []
    if not(isinstance(text, str)):
        return None
    for letter in text.lower():
        if letter.isalpha() and letter != 'º':
            token_list.append(letter)
    return token_list

    """
    Split a text into tokens.

    Convert the tokens into lowercase, remove punctuation, digits and other symbols

    Args:
        text (str): A text

    Returns:
        list[str] | None: A list of lower-cased tokens without punctuation

    In case of corrupt input arguments, None is returned
    """


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    freq_dict = {}
    if not(isinstance(tokens, list)):
        return None
    for letter in tokens:
        if not(isinstance(letter, str)):
            return None
        freq_dict[letter] = tokens.count(letter) / len(tokens)
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
    language_profile = {}
    if not(isinstance(language, str)) or not(isinstance(text, str)):
        return None
    language_profile['name'] = language
    language_profile['freq'] = calculate_frequencies(tokenize(text))
    return language_profile

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
    if not(isinstance(predicted, list)) or not(isinstance(actual, list)):
        return None
    if len(predicted) != len(actual):
        return None
    mse = 0
    n = len(predicted)
    for elem in range(n):
        mse += (actual[elem] - predicted[elem]) ** 2 / n
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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if not('freq' in profile_to_compare) or not('freq' in unknown_profile):
        return None
    if not('name' in profile_to_compare) or not('name' in unknown_profile):
        return None
    if not(isinstance(profile_to_compare['name'], str)) or not(isinstance(profile_to_compare['freq'], dict)):
        return None
    for prof in [unknown_profile['freq'], profile_to_compare['freq']]:
        for elem, freq in prof.items():
            if not(isinstance(elem, str)) or not(isinstance(freq, float) or isinstance(freq, int)):
                return None
    first_language_prof = unknown_profile['freq']
    second_language_prof = profile_to_compare['freq']
    for letter in first_language_prof.keys():
        if not (letter in second_language_prof.keys()):
            second_language_prof[
                letter] = 0  # приводим словари к одному и тому же количеству букв (на случай, если буква встречается в одном словаре, а в другом её нет)
    for letter in second_language_prof.keys():
        if not (letter in first_language_prof.keys()):
            first_language_prof[letter] = 0
    profile_to_compare_sort, unknown_profile_sort = dict(sorted(first_language_prof.items())), dict(
        sorted(second_language_prof.items()))
    compare_result = calculate_mse(list(profile_to_compare_sort.values()), list(unknown_profile_sort.values()))
    return compare_result


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
    if not(isinstance(unknown_profile, dict)) or not(isinstance(profile_1, dict)) or not(isinstance(profile_2, dict)):
        return None
    first_metric = compare_profiles(unknown_profile, profile_1)
    second_metric = compare_profiles(unknown_profile, profile_2)
    if first_metric == second_metric:
        languages = [profile_1['name'], profile_2['name']]
        return sorted(languages)[0]
    elif first_metric < second_metric:
        return profile_1['name']
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
