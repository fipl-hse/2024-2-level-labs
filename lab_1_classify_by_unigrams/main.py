"""
Lab 1.

Language detection
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

def tokenize(text: str) -> list[str] | None:
    from lab_1_classify_by_unigrams.main import tokenize
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

    tokens = []

    for token in text:
        if token.isalpha():
           tokens.append(token.lower())

    return tokens



def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    if not isinstance(tokens, list):
        return None

    frequency_dict = {}

    for letter in tokens:
        if letter.isalpha():
           if letter in frequency_dict:
            frequency_dict[letter] += 1
           else:
            frequency_dict[letter] = 1

    return frequency_dict

    """
    Calculate frequencies of given tokens.

    Args:
        tokens (list[str] | None): A list of tokens

    Returns:
        dict[str, float] | None: A dictionary with frequencies

    In case of corrupt input arguments, None is returned
    """


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
    if not (isinstance(language, str) and isinstance(text, str)):
        return None

    frequency_dict = calculate_frequencies(tokenize(text))

    return {'name': language,
            'freq': frequency_dict}


