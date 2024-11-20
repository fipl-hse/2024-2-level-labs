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
    with open("assets/secrets/secret_4.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    stopwords = open_files()[1]
    documents = open_files()[0]
    tokenizer = Tokenizer(stopwords)
    test_doc = '"Мой кот Вектор по утрам приносит мне тапочки, а по вечерам' \
               ' мы гуляем с ним на шлейке во дворе.' \
               ' Вектор забавный и храбрый. Он не боится собак!'
    tokenized_test_doc = tokenizer.tokenize(test_doc)
    print(tokenized_test_doc)
    test_docs = ['Векторы используются для поиска релевантного документа. Давайте научимся,'
                 ' как их создавать и использовать!','Мой кот Вектор по утрам приносит мне тапочки,'
                 ' а по вечерам мы гуляем с ним на шлейке во дворе.'
                 ' Вектор забавный и храбрый.'
                 ' Он не боится собак!', 'Котёнок, которого мы нашли во дворе,'
                 ' очень забавный и пушистый.'
                 ' По утрам я играю с ним в догонялки перед работой.',
                 'Моя собака думает, что её любимый'
                 ' плед — это кошка. Просто он очень пушистый и мягкий.'
                 ' Забавно наблюдать, как они спят'
                 ' вместе!']
    tokenized_test_docs = tokenizer.tokenize_documents(test_docs)
    print(tokenized_test_docs)

    tokenized_docs = tokenizer.tokenize_documents(documents)
    if not tokenized_docs:
        return
    vectorizer = Vectorizer(tokenized_docs)
    searchengine = BasicSearchEngine(vectorizer,tokenizer)
    searchengine.index_documents(documents)
    secret_tokens = text.split(', ')
    secret_vector = tuple(float(token) for token in secret_tokens)
    print(secret_vector)
    secret = vectorizer.vector2tokens(secret_vector)
    print(secret)
    secret_revealed = searchengine.retrieve_vectorized(secret_vector)
    print(secret_revealed)

    searchengine_pro = SearchEngine(vectorizer, tokenizer)

    result = '???'
    assert result, "Result is None"


if __name__ == "__main__":
    main()
