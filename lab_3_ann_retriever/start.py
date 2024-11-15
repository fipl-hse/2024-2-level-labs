"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import (AdvancedSearchEngine, BasicSearchEngine, KDTree, NaiveKDTree,
                                      SearchEngine, Tokenizer, Vectorizer)


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
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    secret_vector = tuple(float(item) for item in text.split(', '))
    result = ':('
    query = 'Нижний Новгород'
    stopwords = open_files()[1]
    tokenizer = Tokenizer(stopwords)
    documents = open_files()[0]
    tokenized_documents = tokenizer.tokenize_documents(documents)
    # print(tokenized_documents)
    vectorizer = Vectorizer(tokenized_documents)
    vectorizer.build()
    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(documents)
    # print(vectorizer.vector2tokens(secret_vector))
    # print(knn_retriever.retrieve_vectorized(secret_vector))
    naive_tree = NaiveKDTree()
    vectors = knn_retriever._document_vectors
    naive_tree.build(vectors)
    query_vector = vectorizer.vectorize(tokenizer.tokenize(query))
    naive_tree.query(query_vector)
    naive_kdtree_retriever = SearchEngine(vectorizer, tokenizer)
    naive_kdtree_retriever.index_documents(documents)
    print(naive_kdtree_retriever.retrieve_relevant_documents(query))
    kdtree_retriever = AdvancedSearchEngine(vectorizer, tokenizer)
    kdtree_retriever.index_documents(documents)
    vectors = kdtree_retriever._document_vectors
    tree = KDTree()
    tree.build(vectors)
    tree.query(query_vector)
    print(kdtree_retriever.retrieve_relevant_documents(query, 3))
    assert result, "Result is None"


if __name__ == "__main__":
    main()
