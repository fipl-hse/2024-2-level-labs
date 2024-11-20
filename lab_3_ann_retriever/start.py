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
    return documents, stopwords


def main() -> None:
    """
    Launch an implementation.
    """
    #with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        #text = text_file.read()
    documents, stopwords = open_files()
    query = 'Нижний Новгород'
    tokenize = Tokenizer(stopwords)
    corpus = tokenize.tokenize_documents(documents)
    if not isinstance(corpus, list):
        return
    vectorize = Vectorizer(corpus)
    vectorize.build()
    knn_retriever = BasicSearchEngine(vectorize, tokenize)
    knn_retriever.index_documents(documents)
    basic_search = knn_retriever.retrieve_relevant_documents(query, 3)
    print(basic_search)
    naive_kd_tree_retriever = SearchEngine(vectorize, tokenize)
    naive_kd_tree_retriever.index_documents(documents)
    result = naive_kd_tree_retriever.retrieve_relevant_documents(query, 1)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
