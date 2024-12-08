"""
Laboratory Work #4 starter.
"""
from lab_4_retrieval_w_clustering.main import (
    ClusteringSearchEngine,
    DocumentVectorDB,
    get_paragraphs,
    VectorDBSearchEngine,
)

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable


def open_files() -> tuple[list[str], list[str]]:
    """
    # stubs: keep.

    Open files.

    Returns:
        tuple[list[str], list[str]]: Documents and stopwords.
    """
    paths_to_texts = [
        "assets/texts/Master_and_Margarita_chapter1.txt",
        "assets/texts/Master_and_Margarita_chapter2.txt",
        "assets/texts/Master_and_Margarita_chapter3.txt",
        "assets/texts/Master_and_Margarita_chapter4.txt",
        "assets/texts/Master_and_Margarita_chapter5.txt",
        "assets/texts/Master_and_Margarita_chapter6.txt",
        "assets/texts/Master_and_Margarita_chapter7.txt",
        "assets/texts/Master_and_Margarita_chapter8.txt",
        "assets/texts/Master_and_Margarita_chapter9.txt",
        "assets/texts/Master_and_Margarita_chapter10.txt",
        "assets/texts/War_and_Peace_chapter1.txt",
        "assets/texts/War_and_Peace_chapter2.txt",
        "assets/texts/War_and_Peace_chapter3.txt",
        "assets/texts/War_and_Peace_chapter4.txt",
        "assets/texts/War_and_Peace_chapter5.txt",
        "assets/texts/War_and_Peace_chapter6.txt",
        "assets/texts/War_and_Peace_chapter7.txt",
        "assets/texts/War_and_Peace_chapter8.txt",
        "assets/texts/War_and_Peace_chapter9.txt",
        "assets/texts/War_and_Peace_chapter10.txt",
    ]
    documents = []
    for path in sorted(paths_to_texts):
        with open(path, "r", encoding="utf-8") as file:
            documents.append(file.read())
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    return documents, stopwords


def main() -> None:
    """
    Launch an implementation.
    """
    documents, stopwords = open_files()
    document = ''.join(documents)
    corpus_paragraphs = get_paragraphs(document)
    corpus_paragraphs.pop(-1)
    doc_db = DocumentVectorDB(stopwords)
    doc_db.put_corpus(corpus_paragraphs)
    vector_db = VectorDBSearchEngine(doc_db)
    query = "Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления"
    relevant_docs_k_near = vector_db.retrieve_relevant_documents(query, 3)
    print(relevant_docs_k_near)
    cluster_search_engine = ClusteringSearchEngine(doc_db, 5)
    result = cluster_search_engine.retrieve_relevant_documents(query, 5)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
