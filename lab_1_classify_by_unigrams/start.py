"""
Module for starting language profile creation.
"""

from typing import Optional
from main import create_language_profile

def main() -> None:
    """
    Launches the implementation of language profile creation and validation.
    """
    try:
        with open("assets/texts/en.txt", "r", encoding="utf-8") as file_to_read_en:
            en_text = file_to_read_en.read()
        '''
        with open("assets/texts/de.txt", "r", encoding="utf-8") as file_to_read_de:
            de_text = file_to_read_de.read()

        with open("assets/texts/unknown.txt", "r", encoding="utf-8") as file_to_read_unk:
            unknown_text = file_to_read_unk.read()
        '''

        result = create_language_profile('english', en_text)
        print(result)
        assert result is not None, "Detection result is None"

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()