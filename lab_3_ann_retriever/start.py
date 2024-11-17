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
    documents = tuple_file[0]
    doc = documents[0]
    stopwords = tuple_file[1]

    tokenizer = m.Tokenizer(stopwords)
    tokenized_doc = tokenizer.tokenize(doc)
    if not tokenized_doc:
        return None
    tokenized_docs = tokenizer.tokenize_documents(documents)
    if not tokenized_docs:
        return None

    vectorizer = m.Vectorizer(tokenized_docs)
    vectorizer.build()

    query_vector = vectorizer.vectorize(tokenized_doc)
    doc_dist = []
    for doc in tokenized_docs:
        doc_vector = vectorizer.vectorize(doc)
        dist = m.calculate_distance(query_vector, doc_vector)
        doc_dist.append(dist)

    knn_retriever = m.BasicSearchEngine(vectorizer, tokenizer)
    knn_retriever.index_documents(documents)
    knn_result = knn_retriever.retrieve_relevant_documents("Нижний Новгород", 1)

    print(vectorizer.vector2tokens(query_vector))
    print(knn_retriever.retrieve_vectorized(query_vector))

    secret = tuple(text)
    print(vectorizer.vector2tokens(secret))

    naive_tree = m.NaiveKDTree()
    vectors = [(0.0, 0.0, 0.094), (0.061, 0.121, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    naive_tree.build(vectors)
    query_vector = (0.0, 0.0, 0.094)
    naive_tree.query(query_vector)

    naive_kdtree_retriever = m.SearchEngine(vectorizer, tokenizer)
    naive_kdtree_retriever.index_documents(documents)
    naive_kdtree_result = naive_kdtree_retriever.retrieve_relevant_documents("Нижний Новгород")

    result = (knn_result, naive_kdtree_result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
