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
    if isinstance(text, str):
        prepared_text = text.lower()
        tokens = []
        for i in prepared_text:
            if i.isalpha():
                tokens.append(i)
        return tokens
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
    if isinstance(tokens, list):
        if isinstance(tokens[0], str):
            frequency = {}
            tokens_number = len(tokens)
            for letter in tokens:
                if isinstance(letter, str):
                    letter_number = tokens.count(letter)
                    letter_index = letter_number/tokens_number
                    letter_dict = {letter: letter_index}
                    frequency.update(letter_dict)
            return frequency
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
    if isinstance(text, str) and isinstance(language, str):
        freq_dict = calculate_frequencies(tokenize(text))
        language_profile = {"name": language,
                            "freq": freq_dict
                            }
        return language_profile
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
    if isinstance(predicted, list) and isinstance(actual, list):
        if len(actual) == len(predicted):
            summa = 0
            n = len(actual)
            for i in range(0, n):
                a = float(actual[i])
                p = float(predicted[i])
                sqw_dif = float((a - p)**2)
                summa += sqw_dif
            mse = summa/n
            return mse
    return None


def compare_profiles(unknown_profile: dict[str, str | dict[str, float]],
                     profile_to_compare: dict[str, str | dict[str, float]],) -> float | None:
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
    if isinstance(unknown_profile, dict) and isinstance(profile_to_compare, dict):
        if len(unknown_profile) == 2 and len(profile_to_compare) == 2:
            unknown_freq = unknown_profile.get("freq")
            compare_freq = profile_to_compare.get("freq")
            for tok in unknown_freq:
                if tok not in compare_freq:
                    new_tok = {tok: 0.0}
                    compare_freq.update(new_tok)
            compare_freq_l = sorted(compare_freq.items())
            compare_freq_new = {}
            for elem in compare_freq_l:
                if len(elem) == 2:
                    token_for_new = {elem[0]: elem[1]}
                    compare_freq_new.update(token_for_new)

            for tok in compare_freq:
                if tok not in unknown_freq:
                    new_tok = {tok: 0.0}
                    unknown_freq.update(new_tok)
            unknown_freq_l = sorted(unknown_freq.items())
            unknown_freq_new = {}
            for elem in unknown_freq_l:
                if len(elem) == 2:
                    token_for_new = {elem[0]: elem[1]}
                    unknown_freq_new.update(token_for_new)
            profile_pr = list(compare_freq_new.values())
            profile_act = list(unknown_freq_new.values())
            mse = calculate_mse(profile_pr, profile_act)
            mse = round(mse, 4)
            return mse
    return None


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
    if (isinstance(unknown_profile, dict)
            and isinstance(profile_1, dict)
            and isinstance(profile_2, dict)):
        mse_1 = compare_profiles(unknown_profile, profile_1)
        mse_2 = compare_profiles(unknown_profile, profile_2)
        if isinstance(mse_1, float) and isinstance(mse_2, float):
            if mse_1 < mse_2:
                language = profile_1.get('name')
            elif mse_2 < mse_1:
                language = profile_2.get('name')
            else:
                lang_list = [profile_1.get('name'), profile_2.get('name')]
                lang = sorted(lang_list)
                language = lang[0]
            return language
    return None


def load_profile(path_to_file: str) -> dict | None:
    """
    Load a language profile.

    Args:
        path_to_file (str): A path to the language profile

    Returns:
        dict | None: A dictionary with at least two keys – name, freq

    In case of corrupt input arguments, None is returned
    """

    if isinstance(path_to_file, str):
        with open(path_to_file, "r", encoding="utf-8") as file_to_read:
            profile_r = json.load(file_to_read)
            profile = dict(profile_r)
        return profile
    return None


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
    if isinstance(profile, dict) and isinstance(profile.get('freq'), dict):
        frequency = profile.get("freq")
        new_profile = {}
        name = {"name": profile.get("name")}
        new_profile.update(name)
        tokens = list(frequency.keys())
        letters = {}
        n_words = profile.get("n_words")[0]
        for elem in tokens:
            if isinstance(elem, str) and len(elem) == 1:
                number = frequency.get(elem)
                let = elem.lower()
                if let not in letters.keys():
                    new_token = {let: number}
                    letters.update(new_token)
                else:
                    freq_add = letters.get(let)
                    sum_freq = number + freq_add
                    new_token = {let: sum_freq}
                    letters.update(new_token)
        for tok in letters:
            freq_index = float(letters.get(tok))/n_words
            letters[tok] = freq_index
        new_profile["freq"] = letters
        return new_profile
    return None


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collect profiles for a given path.

    Args:
        paths_to_profiles (list): A list of strings to the profiles

    Returns:
        list[dict[str, str | dict[str, float]]] | None: A list of loaded profiles

    In case of corrupt input arguments, None is returned
    """
    if isinstance(paths_to_profiles, list):
        list_of_profiles = []
        for name in paths_to_profiles:
            profile = load_profile(name)
            correct_profile = preprocess_profile(profile)
            list_of_profiles.append(correct_profile)
        return list_of_profiles
    return None


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
    if isinstance(unknown_profile, dict) and isinstance(known_profiles, list):
        compared_languages = {}
        for profile in known_profiles:
            if isinstance(profile, dict):
                mse = compare_profiles(unknown_profile, profile)
                lang_mse = {profile.get('name'): mse}
                compared_languages.update(lang_mse)
        data_to_return = list(compared_languages.items())
        return data_to_return
    return None


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Print report for detection of language.

    Args:
        detections (list[tuple[str, float]]): A list with distances for each available language

    In case of corrupt input arguments, None is returned
    """
