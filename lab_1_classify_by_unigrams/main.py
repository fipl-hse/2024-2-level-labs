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
    tokens = []
    for symbol in text:
        if symbol.isalpha():
            tokens.append(symbol.lower())
        continue
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
    if not (isinstance(tokens, list) and all(isinstance(token, str) for token in tokens)):
        return None
    dict_freq = {}
    for token in tokens:
        if token in dict_freq:
            dict_freq[token] += 1
        else:
            dict_freq[token] = 1
    for token, count in dict_freq.items():
        dict_freq[token] /= len(tokens)
    return dict_freq


def create_language_profile(language: str, text: str) ->\
        dict[str, str | dict[str, float]] | None:
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
    if not isinstance(freq_dict, dict):
        return None
    return {'name': language,
            'freq': freq_dict}


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
    if not (isinstance(predicted, list) and isinstance(actual, list)
            and len(predicted) == len(actual)):
        return None
    summation = 0
    all_number = len(actual)
    for i in range(0, all_number):
        difference = actual[i] - predicted[i]
        squared_difference = difference ** 2
        summation = summation + squared_difference
    return summation / all_number


def compare_profiles(unknown_profile: dict[str, str | dict[str, float]],
                     profile_to_compare: dict[str, str | dict[str, float]],) -> float | None:
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
    if ('name' not in unknown_profile or 'freq' not in unknown_profile or
            'name' not in profile_to_compare or 'freq' not in profile_to_compare):
        return None
    profile_1 = set(unknown_profile['freq'])
    profile_2 = set(profile_to_compare['freq'])
    union_set = profile_1.union(profile_2)
    act_freq = []
    pred_freq = []
    for token in union_set:
        act_freq.append(unknown_profile['freq'].get(token, 0))
        pred_freq.append(profile_to_compare['freq'].get(token, 0))
    return calculate_mse(pred_freq, act_freq)


def detect_language(unknown_profile: dict[str, str | dict[str, float]],
                    profile_1: dict[str, str | dict[str, float]],
                    profile_2: dict[str, str | dict[str, float]],) -> str | None:
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
    if not (isinstance(unknown_profile, dict) and isinstance(profile_1, dict)
            and isinstance(profile_2, dict) and isinstance(profile_1["name"], str)
            and isinstance(profile_2["name"], str)):
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 is None or mse_2 is None:
        return None
    if mse_1 < mse_2:
        return str(profile_1['name'])
    if mse_1 > mse_2:
        return str(profile_2['name'])
    if mse_1 == mse_2:
        sorted_profiles = sorted([profile_1['name'], profile_2['name']])
        return str(sorted_profiles[0])
    return None


def load_profile() -> dict | None:
    """
    Load a language profile.

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
