"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
from time import time

from lab_3_ann_retriever.main import (AdvancedSearchEngine, BasicSearchEngine, SearchEngine,
                                      Tokenizer, Vectorizer)


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
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    documents, stopwords = open_files()
    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    if not tokenized_docs:
        return
    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()
    vectorizer.save("assets/states/vectorizer_state.json")
    new_vectorizer = Vectorizer(tokenized_docs)
    new_vectorizer.load("assets/states/vectorizer_state.json")
    new_vectorizer.build()

    basic_search_engine = BasicSearchEngine(vectorizer, tokenizer)
    basic_search_engine.index_documents(documents)
    basic_start = time()
    basic_result = basic_search_engine.retrieve_relevant_documents("Нижний Новгород", 3)
    basic_finish = time()
    print(f"Results by BasicSearchEngine: {basic_result} \n Time: {basic_finish - basic_start}")

    search_engine = SearchEngine(vectorizer, tokenizer)
    search_engine.index_documents(documents)
    search_engine.save("assets/states/engine_state.json")
    search_start = time()
    search_result = search_engine.retrieve_relevant_documents("Нижний Новгород", 3)
    search_finish = time()
    print(f"Results by SearchEngine: {search_result} \n Time: {search_finish - search_start}")

    advanced_search_engine = AdvancedSearchEngine(new_vectorizer, tokenizer)
    advanced_search_engine.load("assets/states/engine_state.json")
    advanced_search_engine.index_documents(documents)
    advanced_start = time()
    advanced_result = advanced_search_engine.retrieve_relevant_documents("Нижний Новгород", 3)
    advanced_finish = time()
    print(f"Results by AdvancedSearchEngine: {advanced_result} \n"
          f"Time: {advanced_finish - advanced_start}")

    secret_vector = tuple(float(value) for value in text.split(", "))
    print(f"Secret 5 is unraveled! {basic_search_engine.retrieve_vectorized(secret_vector)}")
    result = advanced_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
