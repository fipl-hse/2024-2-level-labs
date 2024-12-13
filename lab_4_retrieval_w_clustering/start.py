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
    stopwords = open_files()[1]
    documents = open_files()[0]
    paragraphs = []
    for doc in documents:
        paragraphs.extend(get_paragraphs(doc))
    query = """Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления"""
    db = DocumentVectorDB(stopwords)
    db.put_corpus(paragraphs)
    vectordb_search_engine = VectorDBSearchEngine(db)
    search_engine_result = vectordb_search_engine.retrieve_relevant_documents(query, 5)
    print(f"Result returned by SearchEngine: {search_engine_result}\n")
    clustering_search = ClusteringSearchEngine(db)
    sse = clustering_search.calculate_square_sum()
    print(f"SSE: {sse}\n")
    clustering_search_result = clustering_search.retrieve_relevant_documents(query, 5)
    print(f"Result returned by ClusteringSearch: {clustering_search_result}\n")
    clustering_search.make_report(7, 'assets/report.json')
    vectordb_tree_search = VectorDBTreeSearchEngine(db)
    tree_search = vectordb_tree_search.retrieve_relevant_documents(query, 1)
    print(f"Result returned by TreeSearchEngine: {tree_search}\n")
    vectordb_advanced_search = VectorDBAdvancedSearchEngine(db)
    result = vectordb_advanced_search.retrieve_relevant_documents(query, 5)
    print(f"Result returned by AdvancedSearchEngine: {result}\n")
    assert result, "Result is None"


if __name__ == "__main__":
    main()
