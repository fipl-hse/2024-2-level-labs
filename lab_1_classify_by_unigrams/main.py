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
    if not text or not isinstance(text, str):
        return None

    text = text.lower()
    tokens = []

    for i in text:
        if i.isalpha():
            tokens.append(i)

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

    if not isinstance(tokens, list):
        return None

    token_freq = {}

    for token in tokens:
        if not isinstance(token, str):
            return None
        if len(token) > 1:
            return None
        if token not in token_freq:
            token_freq[token] = 1.0
        else:
            token_freq[token] += 1.0

    total_tokens = len(tokens)

    for key, value in token_freq.items():
        token_freq[key] = value / total_tokens


    return token_freq





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
    if not language or not text or not isinstance(language, str):
        return None

    tokens = tokenize(text)
    token_freq = calculate_frequencies(tokens)

    if not isinstance(token_freq, dict):
        return None

    return {
        "name": language,
        "freq": token_freq
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
    if len(predicted) != len(actual):
        return None

    mse = 0

    for i, n in enumerate(predicted):
        mse += (n - actual[i]) ** 2

    return mse / len(predicted)




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

    if (not isinstance(unknown_profile, dict)
            or not isinstance(profile_to_compare, dict)):
        return None
    if ("name" not in unknown_profile
            or "freq" not in unknown_profile
            or "name" not in profile_to_compare
            or "freq" not in profile_to_compare):
        return None
    if (not isinstance(profile_to_compare["name"], str)
            or not isinstance(profile_to_compare["freq"], dict)):
        return None

    all_keys = list(set(unknown_profile["freq"]) | set(profile_to_compare["freq"]))
    unknown_freq_list = []
    freq_list_to_compare = []

    for key in all_keys:
        if not key in unknown_profile["freq"] and isinstance(key, str):
            unknown_freq_list.append(0.0)
        elif isinstance(key, str):
            freq_of_key = unknown_profile["freq"][key]
            unknown_freq_list.append(float(freq_of_key))
        if not key in profile_to_compare["freq"] and isinstance(key, str):
            freq_list_to_compare.append(0.0)
        elif isinstance(key, str):
            freq_of_key = profile_to_compare["freq"][key]
            freq_list_to_compare.append(float(freq_of_key))

    return calculate_mse(unknown_freq_list, freq_list_to_compare)


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
    if not unknown_profile or not profile_2 or not profile_1:
        return None

    mse1 = compare_profiles(unknown_profile, profile_1)
    mse2 = compare_profiles(unknown_profile, profile_2)

    if isinstance(mse1, float) and isinstance(mse2, float):
        if mse1 < mse2 and isinstance(profile_1['name'], str) \
                and isinstance(profile_2['name'], str):
            return profile_1['name']
    if isinstance(profile_2['name'], str):
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
