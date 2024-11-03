"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from main import *


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
    stopwords = open_files()[1]
    # documents = ['Векторы используются для поиска релевантного документа. Давайте научимся, как их создавать и использовать!', 'Мой кот Вектор по утрам приносит мне тапочки, а по вечерам мы гуляем с ним на шлейке во дворе. Вектор забавный и храбрый. Он не боится собак!', 'Котёнок, которого мы нашли во дворе, очень забавный и пушистый. По утрам я играю с ним в догонялки перед работой.', 'Моя собака думает, что её любимый плед — это кошка. Просто он очень пушистый и мягкий. Забавно наблюдать, как они спят вместе!']

    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()

    # vectors = [vectorizer.vectorize(doc) for doc in tokenized_docs]

    search = BasicSearchEngine(vectorizer, tokenizer)
    search.index_documents(documents)

    secret_vector = tuple(float(cor) for cor in text.split(", "))
    print(vectorizer.vector2tokens(secret_vector))
    result = search.retrieve_vectorized(secret_vector)
    print(result)
    assert result, "Result is None"

if __name__ == "__main__":
    main()