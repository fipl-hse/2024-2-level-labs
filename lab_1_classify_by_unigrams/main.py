from typing import List, Dict, Optional


def tokenize(text: str) -> Optional[List[str]]:
    """
    Split a text into tokens.

    Convert the tokens into lowercase, remove punctuation, digits and other symbols.

    Args:
        text (str): A text.

    Returns:
        Optional[List[str]]: A list of lower-cased tokens without punctuation or None.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str):
        return None

    text = text.lower()
    text_of_token = [element for element in text if element.isalpha()]

    return text_of_token


def calculate_frequencies(tokens: Optional[List[str]]) -> Optional[Dict[str, float]]:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (Optional[List[str]]): A list of tokens.

    Returns:
        Optional[Dict[str, float]]: A dictionary with frequencies or None.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(tokens, list) or not all(isinstance(element, str) for element in tokens):
        return None

    number_of_tokens = len(tokens)
    if number_of_tokens == 0:
        return {}

    tokens_cnt = {}
    for letter in tokens:
        tokens_cnt[letter] = tokens_cnt.get(letter, 0) + 1

    tokens_frequency = {symbol: value / number_of_tokens for symbol, value in tokens_cnt.items()}

    return tokens_frequency


def create_language_profile(language: str, text: str) -> Optional[Dict[str, Optional[Dict[str, float]]]]:
    """
    Create a language profile.

    Args:
        language (str): A language.
        text (str): A text.

    Returns:
        Optional[Dict[str, Optional[Dict[str, float]]]]: A dictionary with two keys – name, freq or None.

    In case of corrupt input arguments, None is returned.
    """
    if isinstance(language, str) and isinstance(text, str):
        token_list = tokenize(text)
        frequencies = calculate_frequencies(token_list)

        if frequencies is not None:
            profile_of_language = {'name': language, 'freq': frequencies}
            return profile_of_language

    return None
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
