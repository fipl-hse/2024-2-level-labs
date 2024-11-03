"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
from lab_3_ann_retriever.main import Tokenizer, Vectorizer, BasicSearchEngine


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
    secret_vector = tuple(float(item) for item in text.split(', '))
    result = ':('
    stopwords = open_files()[1]
    tokenizer = Tokenizer(stopwords)
    documents = open_files()[0]
    tokenized_documents = tokenizer.tokenize_documents(documents)
    vectorizer = Vectorizer(tokenized_documents)
    vectorizer.build()
    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(documents)
    print(vectorizer.vector2tokens(secret_vector))
    print(knn_retriever.retrieve_vectorized(secret_vector))
    assert result, "Result is None"


if __name__ == "__main__":
    main()
