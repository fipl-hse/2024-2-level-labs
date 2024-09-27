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
    letters = []
    for letter in text:
        if letter.isalpha():
            letters.append(letter.lower())
    return letters

def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
        Calculate frequencies of given tokens.

        Args:
            tokens (list[str] | None): A list of tokens

        Returns:
            dict[str, float] | None: A dictionary with frequencies

        In case of corrupt input arguments, None is returned
    """
    vocabulary = {}
    lenght = len(tokens)
    tokens_set = list(set(tokens))
    for letter in tokens_set:
        amount_of_appear = tokens.count(letter)
        vocabulary[letter] = float(amount_of_appear/lenght)
    return vocabulary


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
    profile = {}
    frequency = calculate_frequencies(tokenize(text))
    if not frequency:
        return None
    else:
        profile['name'] = language
        profile['freq'] = frequency
        return profile


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
    else:
        n = len(predicted)
    summa = 0
    for i in range(n-1):
        summa += predicted[i]-actual[i]
    mse = (summa**2)/n
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
    unknown_profile_freq = list(unknown_profile["freq"].values())
    profile_to_compare_freq = list(profile_to_compare["freq"].values())
    letters = sorted(list(set(unknown_profile["freq"].keys() | set(profile_to_compare["freq"].keys()))))
    while len(unknown_profile_freq) != len(letters):
        unknown_profile_freq.append(0)
    while len(profile_to_compare_freq) != len(letters):
        profile_to_compare_freq.append(0)
    mse = calculate_mse(predicted=unknown_profile_freq, actual=profile_to_compare_freq)
    return round(mse, 3)

print(compare_profiles(unknown_profile=create_language_profile(language="en", text="ab"), profile_to_compare=create_language_profile(language="en", text="bc")))


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
