"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
from lab_3_ann_retriever.main import SearchEngine, Vectorizer, Tokenizer, BasicSearchEngine

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
    docs_tokenized = tokenizer.tokenize_documents(documents)
    vectorizer = Vectorizer(docs_tokenized)

    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(documents)
    print(knn_retriever.retrieve_relevant_documents(query, 3))

    #searcher = SearchEngine(tokenizer= tokenizer, vectorizer= vectorizer)
    #searcher.index_documents(docs)
    result = knn_retriever.retrieve_relevant_documents(query, 3)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
