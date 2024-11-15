"""
Laboratory Work #3 starter.
"""

from pathlib import Path

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from main import Tokenizer, Vectorizer


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
    return documents, stopwords


def main() -> None:
    """
    Launch an implementation.
    """
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    docs, stops = open_files()
    tokenizer = Tokenizer(stops)
    token_docs_list = tokenizer.tokenize_documents(docs)
    vectorizer = Vectorizer(token_docs_list)
    vectorizer.build()
    result = vectorizer.vectorize(token_docs_list[0])
    vectorizer.load("tests/assets/vectorizer_data.json")
    assert result, "Result is None"
    print(result)


if __name__ == "__main__":
    main()
