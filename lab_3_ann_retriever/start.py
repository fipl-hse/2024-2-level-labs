"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable

from pathlib import Path
from time import time

from lab_3_ann_retriever.main import (
    AdvancedSearchEngine,
    BasicSearchEngine,
    SearchEngine,
    Tokenizer,
    Vectorizer
)


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
    result = None
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    text_elements = text.split(", ")
    float_elements = []
    for element in text_elements:
        float_elements.append(float(element))
    vector_of_elements = tuple(float_elements)

    documents, stopwords = open_files()

    tokenizer = Tokenizer(stopwords)
    token_docs_list = tokenizer.tokenize_documents(documents)

    if isinstance(token_docs_list, list):
        vectorizer = Vectorizer(token_docs_list)
        vectorizer.build()
        secret_question = vectorizer.vector2tokens(vector_of_elements)
        vectorizer.save("assets/states/vectorizer_state.json")
        vectorizer_again = Vectorizer(token_docs_list)
        vectorizer_again.load("assets/states/vectorizer_state.json")

        searcher = BasicSearchEngine(vectorizer, tokenizer)
        searcher.index_documents(documents)
        secret_answer = searcher.retrieve_vectorized(vector_of_elements)
        print(time())
        print(secret_question, secret_answer, sep="\n")
        relevant_docs = searcher.retrieve_relevant_documents("Нижний Новгород", 3)
        print(time())
        for doc in relevant_docs:
            print(doc)
        print()

        better_searcher = SearchEngine(vectorizer, tokenizer)
        better_searcher.index_documents(documents)
        more_relevant_docs = better_searcher.retrieve_relevant_documents("Нижний Новгород")
        print(time())
        for doc in more_relevant_docs:
            print(doc)
        print()
        better_searcher.save("assets/states/engine_state.json")

        best_searcher = AdvancedSearchEngine(vectorizer, tokenizer)
        best_searcher.load("assets/states/engine_state.json")
        most_relevant_docs = best_searcher.retrieve_relevant_documents("Нижний Новгород", 3)
        print(time())
        for doc in more_relevant_docs:
            print(doc)

        result = most_relevant_docs

    assert result, "Result is None"


if __name__ == "__main__":
    main()
