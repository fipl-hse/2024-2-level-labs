"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import Tokenizer


def open_files() -> tuple[list[str], list[str]]:
    """
    # stubs: keep.

    Opens files

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


stopwords = open_files()[1]
tokenizer = Tokenizer(stopwords)
documents = open_files()[0]
tokenized_docs = tokenizer.tokenize_documents(documents)
print(tokenized_docs)


def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    result = tokenized_docs
    assert result, "Result is None"


if __name__ == "__main__":
    main()
