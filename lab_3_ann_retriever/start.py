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

    vectorize = Vectorizer(tokenize.tokenize_documents(documents))
    vectorize.build()


    knn = BasicSearchEngine(vectorize, tokenize)
    knn.index_documents(documents)
    res1 = knn.retrieve_relevant_documents(query, 1)
    print(res1)



    vector = vectorize.vectorize(tokenize.tokenize(documents[0]))

    pre_vect = vectorize.vector2tokens(vector)
    print(pre_vect)

    res = knn.retrieve_vectorized(vector)
    print(res)

    search = SearchEngine(vectorize, tokenize)
    search.index_documents(documents)
    search_res = search.retrieve_relevant_documents(query)
    result = search_res
    print(result)


    assert result, "Result is None"


if __name__ == "__main__":
    main()
