"""
Laboratory Work #4 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from lab_4_retrieval_w_clustering.main import (
    ClusteringSearchEngine,
    DocumentVectorDB,
    get_paragraphs,
    VectorDBAdvancedSearchEngine,
    VectorDBSearchEngine,
    VectorDBTreeSearchEngine,
)


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
    return (documents, stopwords)


def main() -> None:
    """
    Launch an implementation.
    """

    documents = ''.join(open_files()[0])
    stopwords = open_files()[1]

    paragraphs = get_paragraphs(documents)

    db = DocumentVectorDB(stopwords)
    db.put_corpus(paragraphs)

    query = "Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления"

    search_engine = VectorDBSearchEngine(db)
    basic_result = search_engine.retrieve_relevant_documents(query, 1)

    relevant_documents = search_engine.retrieve_relevant_documents(query, 3)

    clustering_engine = ClusteringSearchEngine(db, 5)
    clustering_engine.make_report(3, "./assets/report.json")

    tree_search_engine = VectorDBTreeSearchEngine(db)
    prev_relevant_documents = tree_search_engine.retrieve_relevant_documents(query, 1)

    advanced_search_engine = VectorDBAdvancedSearchEngine(db)
    upd_relevant_documents = advanced_search_engine.retrieve_relevant_documents(query, 5)

    result = basic_result, relevant_documents, prev_relevant_documents, upd_relevant_documents
    assert result, "Result is None"


if __name__ == "__main__":
    main()
