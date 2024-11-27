"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import BasicSearchEngine, SearchEngine, Tokenizer, Vectorizer


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

    documents, stopwords = open_files()
    query = 'Нижний Новгород'
    tokenize = Tokenizer(stopwords)
    k = tokenize.tokenize_documents(documents)
    if not isinstance(k, list):
        return None

    vectorize = Vectorizer(k)
    vectorize.build()


    knn = BasicSearchEngine(vectorize, tokenize)
    knn.index_documents(documents)
    res1 = knn.retrieve_relevant_documents(query, 1)
    if not isinstance(res1, list):
        return None

    if not isinstance(documents, list) or len(documents) == 0 or documents[0] is None:
        return None
    tokens = tokenize.tokenize(documents[0])
    if tokens is None or not isinstance(tokens, list):
        return None
    vector = vectorize.vectorize(tokens)


    if not isinstance(vector, tuple):
        return None

    pre_vect = vectorize.vector2tokens(vector)
    if not isinstance(pre_vect, tuple):
        return None

    res = knn.retrieve_vectorized(vector)

    search = SearchEngine(vectorize, tokenize)
    search.index_documents(documents)
    search_res = search.retrieve_relevant_documents(query)
    result = search_res


    assert result, "Result is None"


if __name__ == "__main__":
    main()
