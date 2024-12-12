

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
import time
from pathlib import Path
from time import time

from lab_3_ann_retriever.main import (
    AdvancedSearchEngine,
    BasicSearchEngine,
    SearchEngine,
    Tokenizer,
    Vectorizer,
)

from lab_3_ann_retriever.main import (
    AdvancedSearchEngine,
    BasicSearchEngine,
    SearchEngine,
    Tokenizer,
    Vectorizer,
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
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    vector_from_text = text.split(", ")
    vector = tuple(float(value) for value in vector_from_text)
    stopwords = open_files()[1]
    documents = open_files()[0]
    tokenizer = Tokenizer(stopwords)
    tokenized_documents = tokenizer.tokenize_documents(documents)
    if tokenized_documents is None:
        result = None
        assert result, "Result is None"
    vectorizer = Vectorizer(tokenized_documents)
    vectorizer.build()
    question = vectorizer.vector2tokens(vector)
    if question is None:
        result = None
        assert result, "Result is None"
    preparing_answer = BasicSearchEngine(vectorizer, tokenizer)
    preparing_answer.index_documents(documents)
    answer = preparing_answer.retrieve_vectorized(vector)
    if answer is None:
        result = None
        assert result, "Result is None"
    print(answer)

    query = "Нижний Новгород"
    search_engine = SearchEngine(vectorizer, tokenizer)
    search_engine.index_documents(documents)
    basic_search = BasicSearchEngine(vectorizer, tokenizer)
    basic_search.index_documents(documents)
    result_engine = search_engine.retrieve_relevant_documents(query)
    print(f"Result returned by SearchEngine: {result_engine}\n")
    start_for_basic = time.time()
    basic_results = basic_search.retrieve_relevant_documents(query, 3)
    if basic_results is None:
        result = None
        assert result, "Result is None"
    print(f"Working time of BasicSearchEngine: {time.time() - start_for_basic}\n")
    if result_engine is None or basic_results is None:
        result = None
        assert result, "Result is None"
    for i, basic_result in enumerate(basic_results):
        print(f"Result #{i + 1} returned by BasicSearchEngine: {basic_result}\n")

    vectorizer.save("assets/states/vectorizer_state.json")
    new_vectorizer = Vectorizer(tokenized_documents)
    new_vectorizer.load("assets/states/vectorizer_state.json")
    search_engine.save("assets/states/engine_state.json")
    advanced_search = AdvancedSearchEngine(new_vectorizer, tokenizer)
    advanced_search.load("assets/states/engine_state.json")
    start_for_advanced = time.time()
    advanced_results = advanced_search.retrieve_relevant_documents(query, 3)
    if advanced_results is None:
        result = None
        assert result, "Result is None"
    print(f"Working time of AdvancedSearchEngine: {time.time() - start_for_advanced}\n")
    for i, advanced_result in enumerate(advanced_results):
        print(f"Result #{i + 1} returned by AdvancedSearchEngine: {advanced_result}\n")
    result = advanced_results
    assert result, "Result is None"


if __name__ == "__main__":
    main()