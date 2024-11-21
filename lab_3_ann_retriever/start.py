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
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    secret_vector = tuple(float(item) for item in text.split(', '))
    query = 'Нижний Новгород'
    stopwords = open_files()[1]
    tokenizer = Tokenizer(stopwords)
    documents = open_files()[0]
    tokenized_documents = tokenizer.tokenize_documents(documents)
    print(tokenized_documents)
    if not tokenized_documents:
        return
    vectorizer = Vectorizer(tokenized_documents)
    vectorizer.build()
    start = time.time()
    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(documents)
    print(knn_retriever.retrieve_relevant_documents(query, 3))
    finish = time.time()
    print(f'BasicSearchEngine time: {finish - start}')
    print(vectorizer.vector2tokens(secret_vector))
    print(knn_retriever.retrieve_vectorized(secret_vector))
    start = time.time()
    naive_kdtree_retriever = SearchEngine(vectorizer, tokenizer)
    naive_kdtree_retriever.index_documents(documents)
    print(naive_kdtree_retriever.retrieve_relevant_documents(query))
    finish = time.time()
    print(f'SearchEngine time: {finish - start}')
    vectorizer.save('assets/states/vectorizer_state.json')
    new_vectorizer = Vectorizer(tokenized_documents)
    new_vectorizer.load('assets/states/vectorizer_state.json')
    naive_kdtree_retriever.save('assets/states/engine_state.json')
    start = time.time()
    kdtree_retriever = AdvancedSearchEngine(new_vectorizer, tokenizer)
    kdtree_retriever.load('assets/states/engine_state.json')
    kdtree_retriever.index_documents(documents)
    result = kdtree_retriever.retrieve_relevant_documents(query, 3)
    print(result)
    finish = time.time()
    print(f'AdvancedSearchEngine time: {finish - start}')
    assert result, "Result is None"


if __name__ == "__main__":
    main()
