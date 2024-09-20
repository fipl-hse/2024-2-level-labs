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
    low = text.lower()
    new_list = list(low)
    clear_list = []
    wrong_cases = list("!?/|\.,';:\"#@()*-+=`~ 1234567890")
    for i in new_list:
        if i not in wrong_cases:
            clear_list.append(i)
    return clear_list


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
        Calculate frequencies of given tokens.

        Args:
            tokens (list[str] | None): A list of tokens

        Returns:
            dict[str, float] | None: A dictionary with frequencies

        In case of corrupt input arguments, None is returned
        """
    frequency = {}
    for letter in tokens:
        if letter.isalpha():
            counter = tokens.count(letter) / len(tokens)
            frequency[letter] = counter
    return frequency


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
    freq = tokenize(text)
    dictionary = calculate_frequencies(freq)
    profile = {
        "name": language,
        "freq": dictionary
    }
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
    length = len(predicted)
    mse = float(sum((actual[i] - predicted[i])**2 for i in range(length)) / length)
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
    from_profile1 = unknown_profile["freq"]
    from_profile2 = profile_to_compare["freq"]
    keys_for_1 = {key for key in from_profile1.keys()}
    keys_for_2 = {key for key in from_profile2.keys()}
    letters_need1 = keys_for_1.difference(keys_for_2)
    letters_need2 = keys_for_2.difference(keys_for_1)

    for i in letters_need1:
        from_profile2.setdefault(i, 0)
    for x in letters_need2:
        from_profile1.setdefault(x, 0)

    sorted1 = dict(sorted(from_profile1.items()))
    sorted2 = {}

    for key in sorted1:
        sorted2[key] = from_profile2.get(key, from_profile1[key])

    #get lists with values from two dictionaries
    list1 = list(sorted1.values())
    list2 = list(sorted2.values())
    conclusion = calculate_mse(list2, list1)
    return round(conclusion, 3)


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

    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 == mse_2:
        languages = []
        languages.append(profile_1["name"])
        languages.append(profile_2["name"])
        sort = sorted(languages)
        return sort[0]
    return min(mse_1, mse_2)


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
