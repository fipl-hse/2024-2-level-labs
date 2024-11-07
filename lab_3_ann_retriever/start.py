"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
import lab_3_ann_retriever.main as m


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
    tuple_file = open_files()
    stopwords = tuple_file[1]
    tokenizer = m.Tokenizer(stopwords)
    docs = tuple_file[0]
    doc = docs[0]
    tokenized_doc = tokenizer.tokenize(doc)
    print(tokenized_doc)
    tokenized_docs = tokenizer.tokenize_documents(docs)
    print(tokenized_docs)
    result = tokenizer
    assert result, "Result is None"


if __name__ == "__main__":
    main()
