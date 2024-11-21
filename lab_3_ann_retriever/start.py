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

    documents, stopwords = open_files()

    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    if tokenized_docs is None:
        return
    print('Tokenized documents: ', tokenized_docs)

    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()

    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(documents)

    tokenized_doc = tokenizer.tokenize(documents[0])
    if tokenized_doc is None:
        return
    vector = vectorizer.vectorize(tokenized_doc)
    if vector is None:
        return
    print('Vector: ', vector)
    token_vector = vectorizer.vector2tokens(vector)
    print('Token vector: ', token_vector)
    knn_retriever.retrieve_vectorized(vector)

    search_engine = SearchEngine(vectorizer, tokenizer)
    search_engine.index_documents(documents)

    query = 'Нижний Новгород'
    result = search_engine.retrieve_relevant_documents(query)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
