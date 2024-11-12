"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vectorizer


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
    #with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        #text = text_file.read()
    documents, stopwords = open_files()
    tokenize = Tokenizer(stopwords)
    vectorize = Vectorizer(tokenize.tokenize_documents(documents))
    vectorize.build()
    knn_retriever = BasicSearchEngine(vectorize, tokenize)
    knn_retriever.index_documents(documents)
    tokenize.tokenize(documents[0])
    vector = vectorize.vectorize(tokenize.tokenize(documents[0]))
    print(vectorize.vector2tokens(vector))
    result = knn_retriever.retrieve_vectorized(vector)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
