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
    text = text.replace("º", "")
    cleaned_text = ""
    for symbol in text:
        if symbol.isalpha():
            cleaned_text += symbol
    tokens = []
    for token in cleaned_text:
        tokens.append(token)
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

    if not isinstance(tokens, list) or len(tokens) == 0:
        return None
    abs_freq = {}
    total_elements = len(tokens)
    for token in tokens:
        if not isinstance(token, str):
            return None
        if token in abs_freq:
            abs_freq[token] += 1
        else:
            abs_freq[token] = 1
    rel_freq = {element: freq / total_elements for element, freq in abs_freq.items()}
    return rel_freq


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
    preprocessed_text = tokenize(text)
    relative_frequency = calculate_frequencies(preprocessed_text)
    if (not isinstance(relative_frequency, dict)
            or not all(isinstance(key, str) for key in relative_frequency)
            or not all(isinstance(value, float) for value in relative_frequency.values())):
        return None
    return {"name": language, "freq": relative_frequency}


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
    if not isinstance(predicted, list) or not isinstance(actual, list) \
            or len(predicted) != len(actual):
        return None
    mse = sum((actual[i] - predicted[i])**2 for i in range(len(actual))) / len(actual)
    return float(mse)


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
    if 'name' not in unknown_profile or 'name' not in profile_to_compare:
        return None
    if not all(isinstance(key, str) for key in unknown_profile) \
            or not all(isinstance(key, str) for key in profile_to_compare) \
            or not isinstance(unknown_profile['freq'], dict) \
            or not isinstance(profile_to_compare['freq'], dict):
        return None

    all_freq_keys = set(unknown_profile['freq']).union(set(profile_to_compare['freq']))
    new_unknown_profile = []
    new_profile_to_compare = []
    for key in all_freq_keys:
        if key in unknown_profile["freq"]:
            new_unknown_profile.append(unknown_profile['freq'][key])
        else:
            new_unknown_profile.append(0)
        if key in profile_to_compare["freq"]:
            new_profile_to_compare.append(profile_to_compare['freq'][key])
        else:
            new_profile_to_compare.append(0)
    return calculate_mse(new_unknown_profile, new_profile_to_compare)


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
    if not all(isinstance(profile, dict) for profile in [unknown_profile, profile_1, profile_2])\
            or not all(isinstance(key, str) for key in unknown_profile) \
            or not all(isinstance(key, str) for key in profile_1) \
            or not all(isinstance(key, str) for key in profile_2):
        return None
    mse1 = compare_profiles(unknown_profile, profile_1)
    mse2 = compare_profiles(unknown_profile, profile_2)
    if not isinstance(mse1, float) or not isinstance(mse2, float):
        return None
    if mse1 > mse2:
        return str(profile_2['name'])
    if mse1 < mse2:
        return str(profile_1['name'])
    if str(profile_1['name']) < str(profile_2['name']):
        return str(profile_1['name'])
    return str(profile_2['name'])


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
