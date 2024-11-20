"""
Laboratory Work #3 starter.
"""

from pathlib import Path
from time import time

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
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
    return documents, stopwords


def main() -> None:
    """
    Launch an implementation.
    """
    print(time())
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    text_elements = text.split(", ")
    text_elements = tuple([float(element) for element in text_elements])
    documents, stopwords = open_files()

    tokenizer = Tokenizer(stopwords)
    token_docs_list = tokenizer.tokenize_documents(documents)

    vectorizer = Vectorizer(token_docs_list)
    vectorizer.build()
    secret_question = vectorizer.vector2tokens(text_elements)
    vectorizer.save("assets/states/vectorizer_state.json")
    vectorizer_again = Vectorizer(token_docs_list)
    vectorizer_again.load("assets/states/vectorizer_state.json")

    searcher = BasicSearchEngine(vectorizer, tokenizer)
    searcher.index_documents(documents)
    relevant_docs = searcher.retrieve_relevant_documents("Нижний Новгород", 3)
    print(time())
    secret_answer = searcher.retrieve_vectorized(text_elements)

    better_searcher = SearchEngine(vectorizer, tokenizer)
    better_searcher.index_documents(documents)
    more_relevant_docs = better_searcher.retrieve_relevant_documents("Нижний Новгород")
    print(time())
    better_searcher.save("assets/states/engine_state.json")

    best_searcher = AdvancedSearchEngine(vectorizer, tokenizer)
    best_searcher.load("assets/states/engine_state.json")
    most_relevant_docs = best_searcher.retrieve_relevant_documents("Нижний Новгород", 3)
    print(time())

    result = most_relevant_docs
    print(secret_question, secret_answer, sep="\n")
    for doc in relevant_docs:
        print(doc)
    print()
    for doc in more_relevant_docs:
        print(doc)
    print()
    for doc in result:
        print(doc)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
