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
            tokenized_docs = []
        print(tokenized_docs)

        vectorizer = Vectorizer(tokenized_docs)

        secret_tokenized = tokenizer.tokenize(text)
        if secret_tokenized is None:
            secret_tokenized = []

        secret_vector = vectorizer.vectorize(secret_tokenized)
        if secret_vector is None:
            secret_vector = ()

        tokens_from_vector = vectorizer.vector2tokens(secret_vector)
        print(tokens_from_vector)

        knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
        knn_retriever.index_documents(documents)

        relevant_document = knn_retriever.retrieve_vectorized(secret_vector)
        if relevant_document is None:
            relevant_document = "None"
        print(relevant_document)

        basic_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
        basic_retriever.index_documents(documents)

        kd_tree_retriever = SearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
        kd_tree_retriever.index_documents(documents)

        query = "Нижний Новгород"

        result = tokenized_docs
    assert result, "Result is None"


if __name__ == "__main__":
    main()
