"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable


def tokenize(text):
    if not isinstance(text, str):
        return None
    extra = ',.:?!;%$#№"&()[]{}~’º01‘23456789 -\n><*=@'
    text_only_letters = ''
    for symbol in text:
        if symbol not in extra:
            text_only_letters += symbol
    text_only_letters = list(text_only_letters.lower())
    #print(text_only_letters)
    return text_only_letters


def calculate_frequencies(tokens):
    if not isinstance(tokens, list):
        return None
    if isinstance(tokens, list):
        for element in tokens:
            if type(element) != str:
                return None
    dictionary = {}
    token_number = len(tokens)
    for letter in tokens:
        if letter not in dictionary.keys():
            dictionary[letter] = 1/token_number
        else:
            dictionary[letter] = (dictionary[letter]*token_number + 1)/token_number
    #print(dictionary)
    return dictionary


def create_language_profile(language, text):
    if not isinstance(language, str) or not isinstance(text, str):
        return None
    dictionary = calculate_frequencies(tokenize(text))
    language_profile = {}
    language_profile['name'] = language
    language_profile['freq'] = dictionary
    #print(language_profile)
    return language_profile


def calculate_mse(predicted, actual):
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    """
    Calculate mean squared error between predicted and actual values.

    Args:
        predicted (list): A list of predicted values
        actual (list): A list of actual values

    Returns:
        float | None: The score

    In case of corrupt input arguments, None is returned
    """
    error_score_list = []
    #mse = sum((predicted[i] - actual[i])**2 for i in range(len(actual))) / len(actual)
    for i in range(len(predicted)):
        error_score = (predicted[i] - actual[i])**2
        error_score_list.append(error_score)
    mse = sum(error_score_list)/len(actual)
    return mse


def compare_profiles(unknown_profile, profile_to_compare):
    freq_dict_1 = profile_to_compare['freq']
    keys_of_dict_1 = freq_dict_1.keys()
    freq_dict_2 = unknown_profile['freq']
    keys_of_dict_2 = freq_dict_2.keys()

    # bad input check for unknown_profile:
    if not isinstance(unknown_profile, dict):
        return None
    if isinstance(unknown_profile, dict):
        for i in keys_of_dict_2:
            if not isinstance(i, int):
                return None
            else:
                continue

    # bad input check for profile_to_compare:
    if not isinstance(profile_to_compare, dict):
        return None
    if isinstance(profile_to_compare, dict):
        for i in keys_of_dict_1:
            if not isinstance(i, int):
                return None
            else:
                continue

    for key1 in keys_of_dict_1:
        if key1 in keys_of_dict_2:
            continue
        if key1 not in keys_of_dict_2:
            freq_dict_2[key1] = 0
    for key2 in keys_of_dict_2:
        if key2 in keys_of_dict_1:
            continue
        if key2 not in keys_of_dict_1:
            freq_dict_1[key2] = 0
    #print('Словари с нулевыми частотностями:\n', freq_dict_1, '\n', freq_dict_2)

    sorted_keys_of_dict_1_with_zeros = sorted(freq_dict_1.keys()) #getting the list of sorted letters
    sorted_keys_of_dict_2_with_zeros = sorted(freq_dict_2.keys())

    dict_1_with_zeros = {} #dictionaries where all letters (even with freq zero) are sorted the same way
    dict_2_with_zeros = {}

    for element in sorted_keys_of_dict_1_with_zeros:
        dict_1_with_zeros[element] = freq_dict_1[element]
    for element in sorted_keys_of_dict_2_with_zeros:
        dict_2_with_zeros[element] = freq_dict_2[element]
    #print('Словари с нулевыми частотностями и ОТСОРТИРОВАННЫЕ:\n', dict_1_with_zeros, '\n', dict_2_with_zeros)

    #know the task is to get values from these two dictionaries in the right order (which I've created above)
    values_list_1 = list(dict_1_with_zeros.values())
    values_list_2 = list(dict_2_with_zeros.values())
    #print(values_list_1, '\n', values_list_2)

    mse = calculate_mse(values_list_1, values_list_2)
    print(mse)
    return mse

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
