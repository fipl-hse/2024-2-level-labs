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
    documents = open_files()[0]
    stopwords = open_files()[1]
    paragraphs = get_paragraphs(''.join(documents))
    query = 'Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления'
    db = DocumentVectorDB(stopwords)
    db.put_corpus(paragraphs)
    vector_db_search = VectorDBSearchEngine(db)
    clustering_search = ClusteringSearchEngine(db)
    clustering_search.make_report(5, 'assets/report.json')
    sse = []
    for num in range(1, 15):
        clustering_search = ClusteringSearchEngine(db, n_clusters=num)
        sse.append(clustering_search.calculate_square_sum())
    vectordb_tree_search = VectorDBTreeSearchEngine(db)
    vectordb_advanced_search = VectorDBAdvancedSearchEngine(db)
    all_searches = [vector_db_search, clustering_search, vectordb_tree_search,
                    vectordb_advanced_search]
    result = []
    for method in all_searches:
        neighbours = method.retrieve_relevant_documents(query, 2)
        result.append(neighbours)
    print(result)
    assert result[0], "Result is None"


if __name__ == "__main__":
    main()
