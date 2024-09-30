"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

def clever_round(n: float, decimals: int) -> float :
    multiplier = 10**decimals
    return int(n * multiplier) / multiplier

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

    text = text.lower()
    num = '1234567890'
    symb = ',./?!:;#@*-&<>%'
    tokens = []
    for elem in text:
        if elem not in num and elem not in symb and elem != ' ':
            tokens.append(elem)
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
    token_freq = {}
    if not isinstance(tokens, list):
        return None

    if not tokens:
        return None

    for letter in tokens:
        if not isinstance(letter, str) or len(letter) > 1:
            return None
        if letter not in token_freq:
            token_freq[letter] = 1
        else:
            token_freq[letter] += 1

    for key, value in token_freq.items():
        token_freq[key] = clever_round(value/len(tokens),4)
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
    if not tokenize(text):
        return None
    if not isinstance(language, str):
        return None
    language_profile = {}
    language_profile['name'] = language
    language_profile['freq'] = calculate_frequencies(tokenize(text))
    return language_profile

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
    mse_sum = 0

    if len(predicted) != len(actual):
        return None

    for i in range(len(predicted)):
        mse_sum += (predicted[i] - actual[i]) ** 2

    return clever_round(mse_sum / len(predicted),4)

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
    # shared_tokens = []
    # if len(profile_to_compare) != len(unknown_profile):
    #     return None
    # for elem_known in profile_to_compare:
    #     for elem_who_knows in unknown_profile:
    #         if elem_known == elem_who_knows:
    #             shared_tokens.append(elem_known)

#aaaaa я вообще не понимаю как работает compare_profiles даже теоретически....
'''В данных профилях встречаются следующие символы: ``a``, ``b``, ``c``.
При этом в профиле первого языка их встречаемость равна
``[0.5, 0.5, 0]``, а в профиле второго языка - ``[0, 0.5, 0.5]``.

Приняв встречаемость символов в первом языке за истинные значения и
встречаемость символов во втором языке за предсказанные, мы можем
рассчитать разницу профилей по метрике ``MSE``. Ее значение будет равно
``0.167`` (с округлением до третьего знака).'''
'''Вот откуда берется 0.167 я не понимаю, поэтому как считать тоже :( 
Пробовала общаться с гпт, но не помогло, помогите пожалуйста((((((('''

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