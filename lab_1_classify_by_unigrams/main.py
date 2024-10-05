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
    tokenized_text = []
    if not isinstance(text, str):
        return None
    text = text.lower()
    for element in text:
        if element.isalpha():
            tokenized_text += element
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
    if not isinstance(tokens, list):
        return None
    for element in tokens:
        if not isinstance(element, str):
            return None
    number_of_tokens = len(tokens)
    tokens_quantity = {}
    for letter in tokens:
        if letter not in tokens_quantity:
            tokens_quantity[letter] = 0.0
        tokens_quantity[letter] += 1.0
    tokens_frequency = {}
    for symbol, value in tokens_quantity.items():
        tokens_frequency[symbol] = value/number_of_tokens
    return tokens_frequency


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
    frequency = calculate_frequencies(tokenize(text))
    if not isinstance(frequency, dict):
        return None
    if not isinstance(tokenize(text), list):
        return None
    return {'name': language, 'freq': frequency}


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
    if len(predicted) != len(actual):
        return None
    summa = 0.0
    for index, value in enumerate(actual):
        summa += (value - predicted[index])**2
    mse = summa/len(actual)
    rounded_mse = round(mse, 4)
    return rounded_mse


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
    if not isinstance(unknown_profile, dict):
        return None
    if 'name' not in unknown_profile or not isinstance(unknown_profile['name'], str):
        return None
    if 'freq' not in unknown_profile or not isinstance(unknown_profile['freq'], dict):
        return None
    if (not all(isinstance(k, str)
                and isinstance(v, float) for k, v in unknown_profile['freq'].items())):
        return None

    unknown_freq = unknown_profile.get('freq')
    compared_freq = profile_to_compare.get('freq')
    if not isinstance(compared_freq, dict) or not isinstance(unknown_freq, dict):
        return None
    for letter in unknown_freq:
        if letter not in compared_freq:
            compared_freq[letter] = 0.0
    compared_freq_sort = dict(sorted(compared_freq.items()))
    for letter in compared_freq:
        if letter not in unknown_freq:
            unknown_freq[letter] = 0.0
    unknown_freq_sort = dict(sorted(unknown_freq.items()))
    unknown_list = []
    for element in unknown_freq_sort.values():
        unknown_list.append(element)
    compared_list = []
    for element in compared_freq_sort.values():
        compared_list.append(element)
    return calculate_mse(unknown_list, compared_list)



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
            or not isinstance(profile_1['name'], str)
            or not isinstance(profile_2['name'], str)
            or not all(isinstance(k, str)
                       and isinstance(v, float) for k, v in unknown_profile['freq'].items())):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_2 is None or mse_1 is None:
        return None
    answer = ''
    if mse_1 < mse_2:
        answer = profile_1['name']
        if not isinstance(answer, str):
            return None
    if mse_2 < mse_1:
        answer = profile_2['name']
        if not isinstance(answer, str):
            return None
    if mse_1 == mse_2:
        names = [profile_1['name'], profile_2['name']]
        names.sort()
        answer = names[0]
        if not isinstance(answer, str):
            return None
    return answer


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
