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
        if symbol in extra:
            continue
        else:
            text_only_letters += symbol

    text_only_letters = list(text_only_letters.lower())
    #print(text_only_letters)
    return text_only_letters


def calculate_frequencies(tokens):
    if not isinstance(tokens, list):
        return None
    if isinstance(tokens, list):
        for element in tokens:
            if not isinstance(element, str):
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
    language_profile = {'name': language, 'freq': dictionary}
    #print(language_profile)
    return language_profile


def calculate_mse(predicted, actual):
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if isinstance(predicted, list):
        for i in predicted:
            if isinstance(i, (float, int)):
                continue
            else:
                return None
        if isinstance(actual, list):
            for i in actual:
                if isinstance(i, (float, int)):
                    continue
                else:
                    return None
    if len(predicted) != len(actual):
        return None
    error_score_list = []
    for i in range(len(predicted)):
        error_score = (predicted[i] - actual[i])**2
        error_score_list.append(error_score)
    mse = sum(error_score_list)/len(actual)
    return mse


def compare_profiles(unknown_profile, profile_to_compare):
    # bad input check for unknown_profile:
    if not isinstance(unknown_profile, dict):
        return None
    if isinstance(unknown_profile, dict):
        for i in unknown_profile['freq'].keys():
            if isinstance(i, str):
                continue
            else:
                return None
        for i in unknown_profile['freq'].values():
            if isinstance(i, (int, float)):
                continue
            else:
                return None

    # bad input check for profile_to_compare:
    if not isinstance(profile_to_compare, dict):
        return None
    if isinstance(profile_to_compare, dict):
        for i in profile_to_compare['freq'].keys():
            if isinstance(i, str):
                continue
            else:
                return None
        for i in profile_to_compare['freq'].values():
            if isinstance(i, (int, float)):
                continue
            else:
                return None
    if len(profile_to_compare) == 2 and len(unknown_profile) == 2:
        freq_dict_2 = profile_to_compare['freq']
        freq_dict_1 = unknown_profile['freq']
        for key2 in freq_dict_2.keys():
            if key2 in freq_dict_1.keys():
                continue
            if key2 not in freq_dict_1.keys():
                freq_dict_1[key2] = 0.0
        for key1 in freq_dict_1.keys():
            if key1 in freq_dict_2.keys():
                continue
            if key1 not in freq_dict_2.keys():
                freq_dict_2[key1] = 0.0
    else:
        return None
    #print('Словари с нулевыми частотностями:\n', freq_dict_1, '\n', freq_dict_2)

    sorted_keys_of_dict_1_with_zeros = sorted(freq_dict_2.keys()) #getting the list of sorted letters
    sorted_keys_of_dict_2_with_zeros = sorted(freq_dict_1.keys())

    dict_2_with_zeros = {} #dictionaries where all letters (even with freq zero) are sorted the same way
    dict_1_with_zeros = {}

    for element in sorted_keys_of_dict_2_with_zeros:
        dict_2_with_zeros[element] = freq_dict_2[element]
    for element in sorted_keys_of_dict_1_with_zeros:
        dict_1_with_zeros[element] = freq_dict_1[element]
    #print('Словари с нулевыми частотностями и ОТСОРТИРОВАННЫЕ:\n', dict_1_with_zeros, '\n', dict_2_with_zeros)

    #now the task is to get values from these two dictionaries in the right order (which I've created above)
    values_list_1 = list(dict_1_with_zeros.values())
    values_list_2 = list(dict_2_with_zeros.values())

    score = calculate_mse(values_list_1, values_list_2)
    return score

def detect_language(unknown_profile, profile_1, profile_2):
    # bad input check for unknown_profile:
    if not isinstance(unknown_profile, dict):
        return None
    if isinstance(unknown_profile, dict):
        for i in unknown_profile['freq'].keys():
            if isinstance(i, str):
                continue
            else:
                return None
        for i in unknown_profile['freq'].values():
            if isinstance(i, (int, float)):
                continue
            else:
                return None
    # bad input check for profile_1:
    if not isinstance(profile_1, dict):
        return None
    if isinstance(profile_1, dict):
        for i in profile_1['freq'].keys():
            if isinstance(i, str):
                continue
            else:
                return None
        for i in profile_1['freq'].values():
            if isinstance(i, (int, float)):
                continue
            else:
                return None
    # bad input check for profile_2:
    if not isinstance(profile_2, dict):
        return None
    if isinstance(profile_2, dict):
        for i in profile_2['freq'].keys():
            if isinstance(i, str):
                continue
            else:
                return None
        for i in profile_2['freq'].values():
            if isinstance(i, (int, float)):
                continue
            else:
                return None

    mse_1_and_unknown = compare_profiles(unknown_profile, profile_1)
    mse_2_and_unknown = compare_profiles(unknown_profile, profile_2)
    if mse_1_and_unknown < mse_2_and_unknown:
        return profile_1['name']
    if mse_2_and_unknown < mse_1_and_unknown:
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
