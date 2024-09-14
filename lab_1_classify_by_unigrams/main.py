"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

# подумать про None + откуда берется language(3) + dict comprehensions? + округлить мсе
def tokenize(text: str) -> list[str] | None:
    text = text.lower
    tokenized_list = []
    for i in text():
        if i.isalpha():
            tokenized_list += i
    en_text = tokenized_list
    return en_text

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
    frequency_dict = {}
    for i in tokens:
        frequency_dict[i] = tokens.count(i) / len(tokens)
    return frequency_dict

"""
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
"""


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    lang_profile_dict = {'name': language, 'freq': calculate_frequencies(tokenize(text))}
    return lang_profile_dict


"""
    Create a language profile.

    Args:
        language (str): A language
        text (str): A text

    Returns:
        dict[str, str | dict[str, float]] | None: A dictionary with two keys – name, freq

    In case of corrupt input arguments, None is returned
 """


def calculate_mse(predicted: list, actual: list) -> float | None:
    sum_diff = 0
    for i, k in enumerate(actual):
        difference_between_values = (k - predicted[i]) ** 2
        sum_diff += difference_between_values
    mse = sum_diff / len(predicted)
    return mse


"""
    Calculate mean squared error between predicted and actual values.

    Args:
        predicted (list): A list of predicted values
        actual (list): A list of actual values

    Returns:
        float | None: The score

    In case of corrupt input arguments, None is returned
    """


def compare_profiles(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_to_compare: dict[str, str | dict[str, float]],
) -> float | None:
    for i in unknown_profile['freq']:
        if i not in profile_to_compare['freq']:
            profile_to_compare['freq'][i] = 0
    for i in profile_to_compare['freq']:
        if i not in unknown_profile['freq']:
            unknown_profile['freq'][i] = 0
    sort_unk = dict(sorted(unknown_profile['freq'].items()))
    sort_comp = dict(sorted(profile_to_compare['freq'].items()))
    comp_values_lst = []
    for i in sort_comp.values():
        comp_values_lst.append(i)
    unk_values_lst = []
    for i in sort_unk.values():
        unk_values_lst.append(i)
    return calculate_mse(comp_values_lst, unk_values_lst)



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


def detect_language(
    unknown_profile: dict[str, str | dict[str, float]],
    profile_1: dict[str, str | dict[str, float]],
    profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    if compare_profiles(unknown_profile, profile_1) < compare_profiles(unknown_profile, profile_2):
        closest_profile = profile_1.get('name')
    elif compare_profiles(unknown_profile, profile_1) > compare_profiles(unknown_profile, profile_2):
        closest_profile = profile_2.get('name')
    return closest_profile


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
