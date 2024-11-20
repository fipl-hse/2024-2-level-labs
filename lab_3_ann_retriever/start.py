"""
Laboratory Work #3 starter.
"""

import time
# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

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
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    documents, stopwords = open_files()

    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)

    if not isinstance(tokenized_docs, list):
        result = None
        assert result, "Result is None"
    if not isinstance(tokenized_docs, list):
        result = None
        assert result, "Result is None"
    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()
    docs_vector = [vectorizer.vectorize(tokens) for tokens in tokenized_docs]

    query = 'Нижний Новгород'
    basic_engine = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    basic_engine.index_documents(documents)
    start_time = time.time()
    nearest_basic = basic_engine.retrieve_relevant_documents(query, 1)
    print(f"Результат BasicSearchEngine {nearest_basic} \n "
          f"Время выполнения: {time.time() - start_time}")

    engine = SearchEngine(vectorizer, tokenizer)
    engine.index_documents(documents)
    nearest_engine = engine.retrieve_relevant_documents(query, 1)
    print(f"Результат SearchEngine {nearest_engine} \n")

    file_path_for_vectorizer = "assets/states/vectorizer_state.json"
    vectorizer.save(file_path_for_vectorizer)
    new_vectorizer = Vectorizer(tokenized_docs)
    new_vectorizer.load(file_path_for_vectorizer)

    file_path_for_search_engine = "assets/states/engine_state.json"
    engine.save(file_path_for_search_engine)
    new_engine = AdvancedSearchEngine(new_vectorizer, tokenizer)
    new_engine.load(file_path_for_search_engine)
    start_time_adv = time.time()
    nearest_advanced = new_engine.retrieve_relevant_documents(query, 1)
    print(f"Результат AdvancedSearchEngine {nearest_advanced} \n"
          f"Время выполнения: {time.time() - start_time_adv}")

    result = nearest_advanced
    assert result, "Result is None"


if __name__ == "__main__":
    main()
