"""
Language detection starter
"""
from curses.ascii import isalpha
from collections import  Counter

from lab_1_classify_by_unigrams.main import tokenize, calculate_frequencies, create_language_profile, calculate_mse, compare_profiles, detect_language


#from itertools import count

#from lib2to3.btm_utils import rec_test


# pylint:disable=too-many-locals, unused-argument, unused-variable


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
        en_text = file_to_read_en.read()

    with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
        de_text = file_to_read_de.read()

    with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
        unknown_text = file_to_read_unk.read()



    assert "Detection result is None"

if __name__ == "__main__":
    main()
