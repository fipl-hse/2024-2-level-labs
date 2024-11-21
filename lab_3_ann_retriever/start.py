"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import BasicSearchEngine, SearchEngine, Tokenizer, Vectorizer
import time


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

    documents, stop_words = open_files()
    tokenizer = Tokenizer(stop_words)
    tok_docs = tokenizer.tokenize_documents(documents)
    if tok_docs is None:
        return
    vectorizer = Vectorizer(tok_docs)
    vectorizer.build()

    secret_vector = tuple(float(value.replace(',', '')) for value in text.split())
    secret_tokens = vectorizer.vector2tokens(secret_vector)
    #print(secret_tokens)

    basic_search_start = time.time()
    basic_search_engine = BasicSearchEngine(vectorizer, tokenizer)
    basic_search_engine.index_documents(documents)
    result = basic_search_engine.retrieve_vectorized(secret_vector)
    basic_search_finish = time.time()
    #print(basic_search_start, '\n', result, '\n' basic_search_finish)

    search_start = time.time()
    query = 'Нижний Новгород'
    engine = SearchEngine(vectorizer, tokenizer)
    engine.index_documents(documents)
    result = engine.retrieve_relevant_documents(query, 1)
    search_finish = time.time()
    #print(search_start, '\n', result, '\n'search_finish)

    assert result, "Result is None"


if __name__ == "__main__":
    main()
