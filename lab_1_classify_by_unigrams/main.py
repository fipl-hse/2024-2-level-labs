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

    text = ''.join(letter for letter in text if letter.isalpha()).lower()
    return list(text)



def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if (not isinstance(tokens, list) or
            not all(isinstance(stroka, str) for stroka in tokens) or not tokens):
        return None
    relative_freq = {}
    for char in set(tokens):
        relative_freq[char] = float(tokens.count(char)) / len(tokens)
    return relative_freq


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
    frequencies_of_tokens = calculate_frequencies(tokenize(text))
    if frequencies_of_tokens is not None:
        return {'name': language, 'freq': frequencies_of_tokens}
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
    if (not isinstance(predicted, list) or not isinstance(actual, list)
            or len(predicted) != len(actual)):
        return None
    return float(sum((actual[i] - predicted[i]) ** 2 for i in range(len(actual))) / len(actual))



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
    if 'freq' not in unknown_profile or 'freq' not in profile_to_compare:
        return None
    if (not all(isinstance(key, str) for key in unknown_profile)
            or not all(isinstance(key, str) for key in profile_to_compare)):
        return None
    both_profiles_symbols = set(unknown_profile['freq']).union(set(profile_to_compare['freq']))
    unk_freq = unknown_profile['freq']
    comp_freq = profile_to_compare['freq']
    if not isinstance(unk_freq, dict) or not isinstance(comp_freq, dict):
        return None
    unk_sort = [(unk_freq.get(char, 0.0)) for char in both_profiles_symbols]
    comp_sort = [(comp_freq.get(char, 0.0)) for char in both_profiles_symbols]
    return calculate_mse(unk_sort, comp_sort)


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
    if (not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict)
            or not isinstance(profile_2, dict)):
        return None
    if (not all(isinstance(key, str) for key in unknown_profile)
            or not all(isinstance(key, str) for key in profile_1)
            or not all(isinstance(key, str) for key in profile_2)):
        return None
    first_comp = compare_profiles(unknown_profile, profile_1)
    second_comp = compare_profiles(unknown_profile, profile_2)
    if str(first_comp) < str(second_comp):
        return str(profile_1['name'])
    if str(first_comp) > str(second_comp):
        return str(profile_2['name'])
    if str(profile_2['name']) < str(profile_1['name']):
        return str(profile_2['name'])
    return str(profile_1['name'])


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
