"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
from main import Tokenizer


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
    result = None
    assert result, "Result is None"

    stopwords = open_files()[1]
    tokenizer = Tokenizer(stopwords)

    doc = 'Был снят сюжет о многодетной семье...'
    tokenized_doc = tokenizer.tokenize(doc)

    docs = ['Был снят сюжет...',  'о многодетной семье...']
    tokenized_docs = tokenizer.tokenize_documents(docs)

    result = tokenized_docs
    assert result, "Result is None"


if __name__ == "__main__":
    main()
