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
    documents, stopwords = open_files()
    query = 'Нижний Новгород'

    tokenizer = Tokenizer(stopwords)
    tokenized_documents = tokenizer.tokenize_documents(documents)
    if tokenized_documents is None:
        return

    vectorizer = Vectorizer(tokenized_documents)
    vectorizer.build()
    vectorizer.save('assets/states/vectorizer_state.json')

    start_basic = time()
    basic_engine = BasicSearchEngine(vectorizer, tokenizer)
    basic_engine.index_documents(documents)
    result_basic_engine = basic_engine.retrieve_relevant_documents(query, 1)
    finish_basic = time()
    print(f'Result for BasicSearchEngine: {result_basic_engine}')
    print(f'Time for BasicSearchEngine: {finish_basic - start_basic}')

    start_search_engine = time()
    search_engine = SearchEngine(vectorizer, tokenizer)
    search_engine.index_documents(documents)
    result_search_engine = search_engine.retrieve_relevant_documents(query)
    finish_search_engine = time()
    search_engine.save('assets/states/engine_state.json')
    print(f'Result for SearchEngine: {result_search_engine}')
    print(f'Time for SearchEngine: {finish_search_engine - start_search_engine}')

    new_vectorizer = Vectorizer(tokenized_documents)
    new_vectorizer.load('assets/states/vectorizer_state.json')
    new_vectorizer.build()

    start_advanced = time()
    advanced_engine = AdvancedSearchEngine(new_vectorizer, tokenizer)
    advanced_engine.load('assets/states/engine_state.json')
    advanced_engine.index_documents(documents)
    result_advanced_engine = advanced_engine.retrieve_relevant_documents(query)
    finish_advanced = time()
    print(f'Result for AdvancedSearchEngine: {result_advanced_engine}')
    print(f'Time for AdvancedSearchEngine: {finish_advanced - start_advanced}')

    result = result_advanced_engine
    assert result, "Result is None"


if __name__ == "__main__":
    main()
