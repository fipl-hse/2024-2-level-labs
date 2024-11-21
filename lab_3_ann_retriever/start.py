"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path

from lab_3_ann_retriever.main import SearchEngine, Tokenizer, Vectorizer


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
    query = 'Нижний Новгород'
    tokenizer = Tokenizer(stopwords)
    tokenized_docs = tokenizer.tokenize_documents(documents)
    vectorizer = Vectorizer(tokenized_docs)
    vectorizer.build()


    #knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    #knn_retriever.index_documents(documents)
    #result = knn_retriever.retrieve_relevant_documents(query, 3)

    naive_kdtree_retriever = SearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    naive_kdtree_retriever.index_documents(documents)
    result = naive_kdtree_retriever.retrieve_relevant_documents(query)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
