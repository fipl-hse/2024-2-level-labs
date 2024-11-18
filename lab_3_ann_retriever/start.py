"""
Laboratory Work #3 starter.
"""

from pathlib import Path

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from main import BasicSearchEngine, NaiveKDTree, SearchEngine, Tokenizer, Vectorizer


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
    docs, stops = open_files()

    tokenizer = Tokenizer(stops)
    token_docs_list = tokenizer.tokenize_documents(docs)

    vectorizer = Vectorizer(token_docs_list)
    vectorizer.build()
    vector = vectorizer.vectorize(token_docs_list[0])
    tokens_from_vec = vectorizer.vector2tokens(vector)

    searcher = BasicSearchEngine(vectorizer, tokenizer)
    searcher.index_documents(docs)
    relevant_docs = searcher.retrieve_relevant_documents("Нижний Новгород", 3)
    closest_doc = searcher.retrieve_vectorized(vectorizer.vectorize(tokenizer.tokenize("Нижний Новгород")))

    naive_tree = NaiveKDTree()
    naive_tree.build([
        (0.0, 0.0, 0.094),
        (0.061, 0.121, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0)])
    print(naive_tree.query((-0.01, 0.0, 0.094), 11))

    better_searcher = SearchEngine(vectorizer, tokenizer)
    better_searcher.index_documents(docs)
    more_relevant_docs = better_searcher.retrieve_relevant_documents("Нижний Новгород")

    result = tokens_from_vec
    print(result, closest_doc, sep="\n")
    for doc in relevant_docs:
        print(doc)
    print()
    for doc_tree in more_relevant_docs:
        print(doc_tree)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
