"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable
import json


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
    wrong_cases = list("!?/|&><%\.,';:\"#@()*-+=`~ 1234567890")
    return [symbol for symbol in text.lower() if symbol not in wrong_cases]


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(tokens, list) or not all(isinstance(s, str) for s in tokens):
        return None
    frequency = {}
    for letter in tokens:
        if letter.isalpha():
            counter = tokens.count(letter) / len(tokens)
            frequency[letter] = counter
    return frequency


def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float] | None] | None:
    """
    Create a language profile.

    Args:
        language (str): A language
        text (str): A text

    Returns:
        dict[str, str | dict[str, float]] | None: A dictionary with two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(language, str):
        return None
    if not isinstance(text, str):
        return None
    if tokenize(text) is None:
        return None
    freq = tokenize(text)
    if calculate_frequencies(freq) is None:
        return None
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
    if not isinstance(predicted, list):
        return None
    if not isinstance(actual, list):
        return None
    if len(predicted) != len(actual):
        return None
    mse = 0
    length = len(predicted)
    for index in range(length):
        mse += (actual[index] - predicted[index]) ** 2
    return mse / float(length)


def profiles_bad_input(profile: dict[str, str | dict[str, float]]) -> dict[str, str | dict[str, float]] | None:
    if not isinstance(profile, dict):
        return None
    if len(profile.keys()) != 2:
        return None
    if not all(l in profile for l in ('freq', 'name')):
        return None
    if not isinstance(profile["name"], str):
        return None
    if not isinstance(profile["freq"], dict):
        return None
    if isinstance(profile["freq"], dict):
        for letter in profile["freq"].keys():
            if not isinstance(letter, str):
                return None
        for number in profile["freq"].values():
            if not isinstance(number, float):
                return None
    return profile


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
    if profiles_bad_input(unknown_profile) is None or profiles_bad_input(profile_to_compare) is None:
        return None
    from_profile1 = unknown_profile["freq"]
    from_profile2 = profile_to_compare["freq"]
    keys_for_1 = {key for key in from_profile1.keys()}
    keys_for_2 = {key for key in from_profile2.keys()}
    if keys_for_1.intersection(keys_for_2) == {}:
        return None
    letters_need1 = keys_for_1.difference(keys_for_2)
    letters_need2 = keys_for_2.difference(keys_for_1)

    for i in letters_need1:
        from_profile2.setdefault(i, 0.0)
    for x in letters_need2:
        from_profile1.setdefault(x, 0.0)

    sorted1 = dict(sorted(from_profile1.items()))
    sorted2 = {}

    #common_keys = keys_for_1.intersection(keys_for_2)
    for key in sorted1:
        sorted2[key] = from_profile2.get(key, from_profile1[key])
    list1 = list(sorted1.values())
    list2 = list(sorted2.values())

    conclusion = calculate_mse(list2, list1)
    if not isinstance(conclusion, float):
        return None
    return conclusion


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
    if profiles_bad_input(unknown_profile) is None or profiles_bad_input(profile_1) is None or profiles_bad_input(profile_2) is None:
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 <= mse_2:
        return profile_1["name"]
    return profile_2["name"]


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, 'r', encoding='utf-8') as file_to_read:
        dictionary = json.load(file_to_read)
        if not isinstance(dictionary, dict):
            return None
        return dictionary


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
    if not isinstance(profile, dict):
        return None
    if len(profile.keys()) != 3:
        return None
    if not all(x in profile for x in ['freq', 'name', 'n_words']):
        return None
    if not isinstance(profile['n_words'], list) or len(profile['n_words']) == 0:
        return None
    if not all(isinstance(key, str) for key in profile['freq'].keys()):
        return None
    if not all(isinstance(value, (int, float)) for value in profile['freq'].values()):
        return None
    processed_profile = {'name': profile['name'], 'freq': {}}
    dicty = {}
    for unigram in profile['freq'].keys():
        if unigram.isalpha():
            pass
        if len(unigram) == 1:
            dicty.setdefault(unigram, profile['freq'][unigram])
    letters = list(dicty)
    for letter in letters:
        if not letter.isupper():
            dicty[letter] = dicty.get(letter, 0) + dicty.get(letter.upper(), 0)
            processed_profile['freq'][letter.lower()] = dicty[letter] / profile['n_words'][0]
            if letter.upper() in dicty:
                del dicty[letter.upper()]
    for i in dicty.items():
        if i[0].isupper():
            processed_profile['freq'][i[0].lower()] = i[1] / profile['n_words'][0]
    return processed_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(paths_to_profiles, list):
        return None
    if not all(isinstance(letter, str) for letter in paths_to_profiles):
        return None
    list_for_dictionaries = []
    for name_of_profile in paths_to_profiles:
        if load_profile(name_of_profile) is None:
            return None
        dictionary_unprocessed = load_profile(name_of_profile)
        if not isinstance(dictionary_unprocessed, dict):
            return None
        processed_profile = preprocess_profile(dictionary_unprocessed)
        if processed_profile is None:
            return None
        if not isinstance(processed_profile, dict):
            return None
        if profiles_bad_input(processed_profile) is None:
            return None
        list_for_dictionaries.append(processed_profile)
    return list_for_dictionaries


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
    if profiles_bad_input(unknown_profile) is None:
        return None
    if not isinstance(unknown_profile, dict):
        return None
    if not isinstance(known_profiles, list):
        return None
    sorted_list = []
    for profile in known_profiles:
        mse: float | None = compare_profiles(unknown_profile, profile)
        if mse is None:
            return None
        sorted_list.append((profile["name"], mse))
    if sorted_list:
        return sorted(sorted_list, key=lambda block: (block[1], block[0]))
    return None


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
    if not isinstance(detections, list):
        return None
    for i in detections:
        if not isinstance(i, tuple):
            return None
        if not isinstance(i[0], str) and isinstance(i[1], float):
            return None
    for block in detections:
        print(f'{block[0]}: MSE {block[1]:.5f}')
    return None
