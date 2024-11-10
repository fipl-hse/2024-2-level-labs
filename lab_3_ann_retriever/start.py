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
    docs = tuple_file[0]
    doc = docs[0]
    stopwords = tuple_file[1]

    tokenizer = m.Tokenizer(stopwords)

    tokenized_doc = tokenizer.tokenize(doc)
    if not tokenized_doc:
        return None
    print(tokenized_doc)

    tokenized_docs = tokenizer.tokenize_documents(docs)
    if not tokenized_docs:
        return None
    print(tokenized_docs)

    vectorizer = m.Vectorizer(tokenized_docs)
    vectorizer.build()

    query_vector = vectorizer.vectorize(tokenized_doc)
    doc_dist = []
    for doc in tokenized_docs:
        doc_vector = vectorizer.vectorize(doc)
        dist = m.calculate_distance(query_vector, doc_vector)
        doc_dist.append(dist)

    knn_retriever = m.BasicSearchEngine(vectorizer, tokenizer)
    result = (tokenizer, vectorizer)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
