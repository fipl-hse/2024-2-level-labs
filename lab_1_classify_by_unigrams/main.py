"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

def tokenize(text):

    text = text.lower()
    split_text = ''.join(text)
    token_list = []
    for symbol in split_text:
        if symbol.isalpha():
            token_list.append(symbol)
    return token_list

def calculate_frequencies(token_text):

    freq_dic = {}
    for letter in token_text:
        freq_dic[letter] = token_text.count(letter) / len(token_text)
    return freq_dic

def create_language_profile(language,text):

    language_profile = {"name":language,"freq":calculate_frequencies(tokenize(text))}
    return language_profile

def calculate_mse(predicted, actual):

    result = 0
    for index in range(len(actual)):
        step = (((predicted[index] - actual[index]) ** 2))
        result += step
    return result/len(actual)

def compare_profiles(unknown_profile,profile_to_compare):

    unknown_freq, compare_freq = unknown_profile["freq"], profile_to_compare["freq"]
    unknown_keys, compare_keys = list(unknown_freq.keys()), list(compare_freq.keys())
    all_keys = (unknown_keys + list(set(compare_keys) - set(unknown_keys)))
    all_unknown_profile, all_profile_to_compare = dict.fromkeys(all_keys, 0), dict.fromkeys(all_keys, 0)
    for (key, value) in unknown_freq.items():
        for (key_, value_) in all_unknown_profile.items():
            if key == key_:
                all_unknown_profile[key_] = value
    for (key, value) in compare_freq.items():
        for (key_, value_) in all_profile_to_compare.items():
            if key == key_:
                all_profile_to_compare[key_] = value

    return calculate_mse(list(all_unknown_profile.values()), list(all_profile_to_compare.values()))

def detect_language(unknown_profile,profile_1,profile_2,):

    if compare_profiles(unknown_profile,profile_1) < compare_profiles(unknown_profile,profile_2):
        return profile_1["name"]
    elif compare_profiles(unknown_profile,profile_2) < compare_profiles(unknown_profile,profile_1):
        return profile_2["name"]
    else:
        alpha_sort = [profile_1["name"], profile_2["name"]].sort()
        return alpha_sort[0]

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

profile_1 = {
       'name': 'lang1',
       'freq': {'a': 0.5, 'b': 0.5}
   }
profile_2 = {
       'name': 'lang2',
       'freq': {'b': 0.5, 'c': 0.5}
   }

print(compare_profiles(profile_1,profile_2))