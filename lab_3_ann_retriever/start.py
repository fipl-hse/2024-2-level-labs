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
    return documents[:50], stopwords


def main() -> None:
    """
    Launch an implementation.
    """
    # with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
    # text = text_file.read()

    documents, stopwords = open_files()
    query = 'Нижний Новгород'

    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    if tokenized_docs is None:
        return

    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()

    search_engine = SearchEngine(vectorizer, tokenizer)
    search_engine.index_documents(documents)
    result_engine = search_engine.retrieve_relevant_documents(query)
    print("Result returned by SearchEngine: ", result_engine)

    basic_search_engine = BasicSearchEngine(vectorizer, tokenizer)
    basic_search_engine.index_documents(documents)
    result_basic_engine = search_engine.retrieve_relevant_documents(query)
    print("Result returned by BasicSearchEngine: ", result_basic_engine)

    result = result_engine
    assert result, "Result is None"


if __name__ == "__main__":
    main()
