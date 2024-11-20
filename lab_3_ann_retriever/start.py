"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import Tokenizer, Vectorizer, calculate_distance, BasicSearchEngine


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

    documents = open_files()[0]
    doc = documents[0]
    stopwords = open_files()[1]

    tokenizer = Tokenizer(stopwords)
    tokenized_doc = tokenizer.tokenize(doc)
    tokenized_docs = tokenizer.tokenize_documents(documents)

    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()

    query_vector = vectorizer.vectorize(tokenized_doc)
    doc_dist = []
    for doc in tokenized_docs:
        doc_vector = vectorizer.vectorize(doc)
        dist = calculate_distance(query_vector, doc_vector)
        doc_dist.append(dist)

    knn_retriever = BasicSearchEngine(vectorizer, tokenizer)
    knn_retriever.index_documents(documents)
    result = knn_retriever.retrieve_relevant_documents("Нижний Новгород", 1)
    assert result, "Result is None"
    print(vectorizer.vector2tokens(query_vector))
    print(knn_retriever.retrieve_vectorized(query_vector))
    print(result)


if __name__ == "__main__":
    main()
