"""
Language detection starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable

from main import tokenize
from main import calculate_frequencies
from main import create_language_profile

def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        tokens = tokenize(en_text)
        print(f'Tokens: {tokens}')
    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()

    print(f'Frequencies: {calculate_frequencies(tokens)}')
    print(f'Language profile: {create_language_profile("en", en_text)}')

    result = tokens
    assert result, "Detection result is None"

if __name__ == "__main__":
    main()
