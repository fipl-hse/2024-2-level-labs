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
    low = text.lower()
    symbols = []
    for symbol in low:
        symbols.append(symbol)
    clear_list = []
    wrong_cases = list("!?/|&><%\.,';:\"#@()*-+=`~ 1234567890")
    for i in symbols:
        if i not in wrong_cases:
            clear_list.append(i)
    return clear_list

a = tokenize("Hello, my dear friend! I would like to see you!!! **")


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

print(calculate_frequencies(a))


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
    #mse = float(sum((actual[i] - predicted[i])**2 for i in range(length)) / length)


print(calculate_mse([0.1538, 0.0, 0.0, 0.0769, 0.0769, 0.0769, 0.0, 0.0, 0.0769, 0.0769, 0.0769, 0.1538, 0.2307, 0.0], [0.1666, 0.1666, 0.0333, 0.1333, 0.0, 0.0666, 0.0666, 0.0333, 0.0333, 0.1, 0.0666, 0.0, 0.0666, 0.0666]))

def profiles_bad_input(profile: dict[str, str | dict[str, float]]) -> dict[str, str | dict[str, float]] | None:
    if not isinstance(profile, dict):
        return None
    if len(profile.keys()) != 2:
        return None
    if "name" not in profile and "freq" not in profile:
        return None #example of profile: { "name": "en", "freq": {"g": 0.89, "t": 0.89} }
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

print(profiles_bad_input({
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538, 'm': 0.0769, 'i': 0.0769,
                'a': 0.2307, 's': 0.0769, 'p': 0.1538
            }
        }))
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
    if profiles_bad_input(unknown_profile) == None or profiles_bad_input(profile_to_compare) == None:
        return None
    from_profile1 = unknown_profile["freq"] #here we have dictionaries like ["h": 0.056]
    from_profile2 = profile_to_compare["freq"]
    keys_for_1 = {key for key in from_profile1.keys()} #take all keys from these dictionaries
    keys_for_2 = {key for key in from_profile2.keys()}
    if keys_for_1.intersection(keys_for_2) == {}:
        return None
    letters_need1 = keys_for_1.difference(keys_for_2) #we need to create the rang of keys without value (with 0.0)
    letters_need2 = keys_for_2.difference(keys_for_1)

    for i in letters_need1:
        from_profile2.setdefault(i, 0.0)
    for x in letters_need2:
        from_profile1.setdefault(x, 0.0)

    sorted1 = dict(sorted(from_profile1.items()))
    sorted2 = {}

    for key in sorted1:
        sorted2[key] = from_profile2.get(key, from_profile1[key])
    list1 = list(sorted1.values())
    list2 = list(sorted2.values())
    conclusion = calculate_mse(list2, list1)
    if type(conclusion) is not float and conclusion <= 0:
        return None
    return conclusion

print(compare_profiles({
            'name': 'en',
            'freq': {
                'y': 0.0769, 'n': 0.0769, 'e': 0.0769, 'h': 0.1538, 'm': 0.0769, 'i': 0.0769,
                'a': 0.2307, 's': 0.0769, 'p': 0.1538
            }
        }, {
            'name': 'de',
            'freq': {
                'u': 0.0344, 'e': 0.1724, 'a': 0.0344, 'r': 0.0344, 'i': 0.1034, 'g': 0.0344,
                'b': 0.0344, 't': 0.0344, 'd': 0.0344, 'm': 0.0344, 's': 0.1034, 'h': 0.0689,
                'c': 0.0689, 'v': 0.0344, 'l': 0.1034, 'ü': 0.0344, 'n': 0.0344
            }
        }))

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
    if profiles_bad_input(unknown_profile) == None or profiles_bad_input(profile_1) == None or profiles_bad_input(profile_2) == None:
        return None
    mse_1 = compare_profiles(unknown_profile, profile_1)
    mse_2 = compare_profiles(unknown_profile, profile_2)
    if mse_1 <= mse_2:
        return profile_1["name"]
    else:
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
