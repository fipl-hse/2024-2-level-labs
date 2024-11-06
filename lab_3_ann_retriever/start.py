"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from lab_3_ann_retriever.main import (BasicSearchEngine, Tokenizer, Vectorizer)
from pathlib import Path


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
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    stopwords = open_files()[1]
    documents = open_files()[0]
    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()
    secret_values = text.replace("\n", "").split(", ")
    secret_vector = tuple(float(value) for value in secret_values)
    secret_tokens = vectorizer.vector2tokens(secret_vector)
    if secret_tokens:
        print(secret_tokens)
    search_engine = BasicSearchEngine(vectorizer, tokenizer)
    search_engine.index_documents(documents)
    result = search_engine.retrieve_vectorized(secret_vector)
    if result:
        print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
