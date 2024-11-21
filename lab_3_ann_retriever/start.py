"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vector, Vectorizer


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
    docs = open_files()[0][:50]
    swrds = open_files()[1]
    tkn = Tokenizer(swrds)
    tokdocs = tkn.tokenize_documents(docs)
    if tokdocs is None:
        return None
    print(tokdocs)
    vect = Vectorizer(tokdocs)
    vect.build()
    print(vect.vector2tokens(Vector(text)))
    knn_retriever = BasicSearchEngine(vectorizer=vect, tokenizer=tkn)
    knn_retriever.index_documents(docs)
    print(knn_retriever.retrieve_vectorized(Vector(text)))
    result = 1
    assert result, "Result is None"


if __name__ == "__main__":
    main()
