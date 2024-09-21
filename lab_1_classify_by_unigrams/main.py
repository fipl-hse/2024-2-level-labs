import json

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
    if type(text) == str:
        list_of_letters = []
        for character in text:
            if character.isalpha():
                list_of_letters.append(character.lower())
        return list_of_letters
    else:
        return None


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """
    if isinstance(tokens, list) and all(isinstance(token, str) for token in tokens):
        freq = dict.fromkeys(tokens)
        for letter in freq:
            freq[letter] = tokens.count(letter) / len(tokens)
        return freq
    else:
        return None


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
    if isinstance(language, str) and isinstance(text, str):
        language_profile = {'name': language, 'freq': calculate_frequencies(tokenize(text))}
        return language_profile
    else:
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
    if not isinstance(predicted, list) or not isinstance(actual, list) or not len(actual) == len(predicted):
        return None
    else:
        n = len(actual)
        mse = 0
        for i in range(n):
            mse += ((actual[i] - predicted[i]) ** 2)
        mse /= n
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
    if (not isinstance(unknown_profile, dict) or
            'name' not in unknown_profile or 'freq' not in unknown_profile or
            not isinstance(profile_to_compare, dict) or
            'name' not in profile_to_compare or 'freq' not in profile_to_compare):
        return None
    else:
        unknown_letters_list = list(unknown_profile['freq'].keys())
        compare_letters_list = list(profile_to_compare['freq'].keys())
        all_letters = sorted(list(set(unknown_letters_list + compare_letters_list)))
        unknown_frequencies_list = []
        compare_frequencies_list = []
        for letter in all_letters:
            if letter in unknown_letters_list and letter in compare_letters_list:
                unknown_frequencies_list.append(unknown_profile['freq'][letter])
                compare_frequencies_list.append(profile_to_compare['freq'][letter])
            elif letter in compare_letters_list:
                compare_frequencies_list.append(profile_to_compare['freq'][letter])
                unknown_frequencies_list.append(0)
            else:
                unknown_frequencies_list.append(unknown_profile['freq'][letter])
                compare_frequencies_list.append(0)
        return calculate_mse(compare_frequencies_list, unknown_frequencies_list)


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
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict) or not isinstance(profile_2, dict):
        return None
    else:
        comparison_results = {
            profile_1['name']: compare_profiles(unknown_profile, profile_1),
            profile_2['name']: compare_profiles(unknown_profile, profile_2)
        }
        potential_languages = []
        for name in comparison_results:
            if comparison_results[name] == min(comparison_results.values()):
                potential_languages.append(name)
        return sorted(potential_languages)[0]


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
    else:
        with open(path_to_file, 'r') as profile_file:
            return json.load(profile_file)


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
