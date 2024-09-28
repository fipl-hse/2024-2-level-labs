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
    extra = ',.:?!;%$#№"&()[]{}~’º01‘23456789 -\n><*=@'
    text_only_letters = ''
    for symbol in text:
        if symbol in extra:
            continue
        text_only_letters += symbol

    text_only_letters_list = list(text_only_letters.lower())
    return text_only_letters_list


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
    if isinstance(tokens, list):
        for element in tokens:
            if not isinstance(element, str):
                return None
    dictionary = {}
    token_number = len(tokens)
    for letter in tokens:
        if letter not in dictionary:
            dictionary[letter] = 1/token_number
        else:
            dictionary[letter] = (dictionary[letter]*token_number + 1)/token_number
    return dictionary


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
    dictionary = calculate_frequencies(tokenize(text))
    language_profile = {'name': language, 'freq': dictionary}
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
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if isinstance(predicted, list):
        for i in predicted:
            if isinstance(i, (float, int)):
                continue
            return None
        if isinstance(actual, list):
            for i in actual:
                if isinstance(i, (float, int)):
                    continue
                return None
    if len(predicted) != len(actual):
        return None
    error_score_list = []
    for i, el in enumerate(predicted):
        error_score = (el - actual[i])**2
        error_score_list.append(error_score)
    mse: float = sum(error_score_list) / len(actual)
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
    # bad input check:
    if isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict):
        for k, v in unknown_profile['freq'].items():
            if not isinstance(k, str) or not isinstance(v, (int, float)):
                return None

    if len(profile_to_compare) == 2 and len(unknown_profile) == 2:
        freq_dict_2 = profile_to_compare['freq']
        freq_dict_1 = unknown_profile['freq']
        for key2 in freq_dict_2.keys():
            if key2 in freq_dict_1.keys():
                continue
            freq_dict_1[key2] = 0.0
        for key1 in freq_dict_1.keys():
            if key1 in freq_dict_2.keys():
                continue
            freq_dict_2[key1] = 0.0
    else:
        return None

    sorted_keys_of_dict_1_with_zeros = sorted(freq_dict_2.keys())
    sorted_keys_of_dict_2_with_zeros = sorted(freq_dict_1.keys())

    dict_2_with_zeros = {} #dictionaries where ALL letters are sorted
    dict_1_with_zeros = {}

    for element in sorted_keys_of_dict_2_with_zeros:
        dict_2_with_zeros[element] = freq_dict_2[element]
    for element in sorted_keys_of_dict_1_with_zeros:
        dict_1_with_zeros[element] = freq_dict_1[element]

    #getting values from two dictionaries in the right order
    values_list_1 = list(dict_1_with_zeros.values())
    values_list_2 = list(dict_2_with_zeros.values())

    score = calculate_mse(values_list_1, values_list_2)
    return score


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
    # bad input check:
    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    for k, v in (unknown_profile['freq'].items()
                 and profile_1['freq'].items()
                 and profile_2['freq'].items()):
        if not isinstance(k, str) or not isinstance(v, (int, float)):
            return None

    mse_1_and_unknown = compare_profiles(unknown_profile, profile_1)
    mse_2_and_unknown = compare_profiles(unknown_profile, profile_2)
    if (not isinstance(mse_1_and_unknown, (int, float))
            or not isinstance(mse_2_and_unknown, (int, float))):
        return None
    result = None
    if mse_1_and_unknown < mse_2_and_unknown:
        result = str(profile_1['name'])
    if mse_2_and_unknown < mse_1_and_unknown:
        result = str(profile_2['name'])
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
