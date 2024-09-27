"""
Language detection starter
"""
from curses.ascii import isalpha
from collections import  Counter
#from itertools import count

#from lib2to3.btm_utils import rec_test


# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()
        lowered_eng_text = en_text.lower()
        english_text = "".join(letter for letter in lowered_eng_text if letter.isalpha())
        english_tokens = list(english_text)

    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()
        lowered_de_text = de_text.lower()
        deutch_text = "".join(letter for letter in lowered_de_text if letter.isalpha())
        deutch_tokens = list(deutch_text)

    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()
        lowered_unk_text = unknown_text.lower()
        unk_text = "".join(letter for letter in lowered_unk_text if letter.isalpha())
        unk_tokens = list(unk_text)

        eng_freq = Counter(english_tokens)
        de_freq = Counter(deutch_tokens)
        unk_freq = Counter(unk_tokens)

        english_tokens_number = len(english_tokens)
        deutch_tokens_number = len(deutch_tokens)
        unk_tokens_number = len(unk_tokens)

        def calculate_relative_frequencies(counter: Counter, tokens_num: int) -> dict:
            if tokens_num == 0:
                return {}
            else:
                return {token: count / tokens_num for token, count in counter.items()}

        english_relative_frequency = calculate_relative_frequencies(eng_freq, english_tokens_number)
        deutch_relative_frequency = calculate_relative_frequencies(de_freq, deutch_tokens_number)
        unknown_relative_frequency = calculate_relative_frequencies(unk_freq, unk_tokens_number)

    result = {
        "number of english tokens": english_tokens_number,
        "number of deutch tokens": deutch_tokens_number,
        "number of unknown tokens": unk_tokens_number,
        "english frequencies": english_relative_frequency,
        "deutch frequencies": deutch_relative_frequency,
        "unknown frequencies": unknown_relative_frequency
    }


    assert result["number of english tokens"] > 0 or result["number of deutch tokens"] > 0 or result["number of unknown tokens"] > 0, "Detection result is None"
    print(english_relative_frequency)
    print(deutch_relative_frequency)
    print(unknown_relative_frequency)

if __name__ == "__main__":
    main()
