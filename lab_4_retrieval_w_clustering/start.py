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
    documents = "".join(open_files()[0])
    corpus = get_paragraphs(documents)
    query = "Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления"
    db = DocumentVectorDB(open_files()[1])
    db.put_corpus(corpus)
    vector_search = VectorDBSearchEngine(db)
    vector_result = vector_search.retrieve_relevant_documents(query, 3)
    clustering_search = ClusteringSearchEngine(db)
    clustering_result = clustering_search.retrieve_relevant_documents(query, 5)
    tree_search = VectorDBTreeSearchEngine(db)
    tree_result = tree_search.retrieve_relevant_documents(query, 5)
    advanced_search = VectorDBAdvancedSearchEngine(db)
    result = advanced_search.retrieve_relevant_documents(query, 5)
    for i in range(1, 15):
        cluster_search = ClusteringSearchEngine(db, i)
        print(f"Number of clusters: {i} SSE: {cluster_search.calculate_square_sum()} \n")
    print(f"Results by VectorDBSearchEngine: \n {vector_result} \n"
          f"Results by ClusteringSearchEngine: \n {clustering_result} \n"
          f"Results by VectorDBTreeSearchEngine: \n {tree_result} \n"
          f"Results by VectorDBAdvancedSearchEngine: \n {result}")
    clustering_search.make_report(3, "assets/report.json")
    assert result, "Result is None"


if __name__ == "__main__":
    main()
