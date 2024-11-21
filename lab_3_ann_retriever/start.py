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
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    tokenizer = Tokenizer(open_files()[1])
    tokenized_docs = tokenizer.tokenize_documents(open_files()[0])
    if not isinstance(tokenized_docs, list):
        return None

    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()

    text_tuple = tuple(float(value) for value in text.split(', '))
    secret_tokens = vectorizer.vector2tokens(text_tuple)
    print(secret_tokens)

    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(open_files()[0])

    result = knn_retriever.retrieve_vectorized(text_tuple)
    print(result)
    assert result, "Result is None"

if __name__ == "__main__":
    main()
