"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import Tokenizer


def open_files() -> tuple[list[str], list[str]]:
    """
    # stubs: keep.

    Open files.

    Returns:
        tuple[list[str], list[str]]: Documents and stopwords
    """
    documents = []
    for path in sorted(Path("assets/articles").glob("*.txt")):
        with open(path, "r", encoding="utf-8") as file:
            documents.append(file.read())
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    return (documents, stopwords)


def main() -> None:
    """
    Launch an implementation.
    """
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    documents_and_stopwords = open_files()
    documents = documents_and_stopwords[0]
    stopwords = documents_and_stopwords[1]
    tokenizer = Tokenizer(stopwords)
    documents_tokenized = tokenizer.tokenize_documents(documents)
    if not isinstance(documents_tokenized, list):
        return
    print(f'Demo 1: Documents Tokenization:\n{documents_tokenized[0]}')
    # result = None
    # assert result, "Result is None"


if __name__ == "__main__":
    main()
