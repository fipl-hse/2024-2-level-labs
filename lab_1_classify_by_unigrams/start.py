"""
Language detection starter
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
    en_text_list = []
    en_text_list = text.split()
    words_list = []

    for i in range(len(en_text_list)):
        letters = list(en_text_list[i])
        words_list.append(letters)
        i += 1

    letters_list = []

    for j in range(len(words_list)):
        for i in range(len(words_list[j])):
            if words_list[j][i].isalpha():
                letters_list.append(words_list[j][i])
            i += 1
        j += 1


    letters_list = str(letters_list).lower()


    return letters_list

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
    result = print(tokenize(en_text))
    assert result, "Detection result is None"


if __name__ == "__main__":
    main()
