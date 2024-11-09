"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import AdvancedSearchEngine, SearchEngine, Tokenizer, Vectorizer


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
    # with open("assets/secrets/secret_3.txt", "r", encoding="utf-8") as text_file:
    #     text = text_file.read()
    documents = open_files()[0]
    stopwords = open_files()[1]

    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    if tokenized_docs is None:
        return None
    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()
    vectorizer.save("assets/states/vectorizer_state.json")
    new_vectorizer = Vectorizer(tokenized_docs)
    new_vectorizer.load("assets/states/vectorizer_state.json")

    # naive_kdtree_retriever = SearchEngine(vectorizer=new_vectorizer, tokenizer=tokenizer)
    # naive_kdtree_retriever.index_documents(documents)
    # naive_kdtree_retriever.save("assets/states/engine_state.json")

    kdtree_retriever = AdvancedSearchEngine(new_vectorizer, tokenizer)
    kdtree_retriever.load("assets/states/engine_state.json")
    query = "Нижний Новгород"
    result = kdtree_retriever.retrieve_relevant_documents(query, 3)
    print(result)
    assert result, "Result is None"
    return None


if __name__ == "__main__":
    main()
