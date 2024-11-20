"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vectorizer


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
    with open("assets/secrets/secret_2.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    documents, stopwords = open_files()
    tokenizer = Tokenizer(stopwords)
    documents_tokenized = tokenizer.tokenize_documents(documents)
    if not isinstance(documents_tokenized, list):
        return
    vectorizer = Vectorizer(documents_tokenized)
    vectorizer.build()
    knn_retriever = BasicSearchEngine(vectorizer, tokenizer)
    print('start')
    knn_retriever.index_documents(documents)

    text_vector = tuple(float(distance) for distance in text.split(","))
    text_tokens = vectorizer.vector2tokens(text_vector)
    text_relevant_document = knn_retriever.retrieve_vectorized(text_vector)

    if not isinstance(documents_tokenized, list):
        return
    print(f'Demo 1: Documents Tokenization:\n{documents_tokenized[0]}')
    print(f'Demo 2: Secret #2 Question\n{text_tokens}')
    print(f'Demo 2: Answer\n{text_relevant_document}')
    result = text_relevant_document
    assert result, "Result is None"


if __name__ == "__main__":
    main()
