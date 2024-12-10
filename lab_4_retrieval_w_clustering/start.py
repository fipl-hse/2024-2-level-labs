"""
Laboratory Work #4 starter.
"""
from lab_4_retrieval_w_clustering.main import (
    ClusteringSearchEngine,
    DocumentVectorDB,
    get_paragraphs,
    VectorDBAdvancedSearchEngine,
    VectorDBSearchEngine,
    VectorDBTreeSearchEngine,
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
    return (documents, stopwords)


def main() -> None:
    """
    Launch an implementation.
    """
    documents = ''.join(open_files()[0])
    paragraphs = get_paragraphs(documents)
    stopwords = open_files()[-1]
    query = ("Первый был не кто иной, как Михаил Александрович "
             "Берлиоз, председатель правления")
    db = DocumentVectorDB(stopwords)
    db.put_corpus(paragraphs)
    vector = VectorDBSearchEngine(db)
    result_for_db = vector.retrieve_relevant_documents(query, 3)

    clusters = ClusteringSearchEngine(db)
    result_for_clusters = clusters.retrieve_relevant_documents(query, 5)

    search = VectorDBTreeSearchEngine(db)
    result_for_search = search.retrieve_relevant_documents(query, 5)

    advanced_search = VectorDBAdvancedSearchEngine(db)
    result = (advanced_search.retrieve_relevant_documents(query, 5))

    for i in range(1, 15):
        cluster_search = ClusteringSearchEngine(db, i)
        print(f"Number of clusters: {i}, SSE: {cluster_search.calculate_square_sum()} \n")
    print(f"Results reported by VectorDBSearchEngine: {result_for_db} \n"
          f"Results reported by ClusteringSearchEngine: {result_for_clusters} \n"
          f"Results reported by VectorDBTreeSearchEngine: {result_for_search} \n"
          f"Results reported by VectorDBAdvancedSearchEngine: {result}")
    clusters.make_report(3, 'assets/report.json')

    assert result, "Result is None"


if __name__ == "__main__":
    main()
