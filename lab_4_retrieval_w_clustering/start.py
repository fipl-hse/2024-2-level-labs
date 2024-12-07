"""
Laboratory Work #4 starter.
"""

import json

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
    paragraphs = get_paragraphs(documents)
    stopwords = open_files()[-1]
    query = 'Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления'
    db = DocumentVectorDB(stopwords)
    db.put_corpus(paragraphs)
    vector_search = VectorDBSearchEngine(db)
    result = vector_search.retrieve_relevant_documents(query, 3)
    print(result)
    # for num in range(1, 15):
    #     clustering_search = ClusteringSearchEngine(db, n_clusters=num)
    #     print(clustering_search.calculate_square_sum())
    #     clustering_search.retrieve_relevant_documents(query, 5)
    #     clustering_search.make_report(3, 'assets/report.json')
    #     with open('assets/report.json', 'r', encoding='utf-8') as file_to_read:
    #         state = json.load(file_to_read)
    #     print(state)
    vector_tree_search = VectorDBTreeSearchEngine(db)
    print(vector_tree_search.retrieve_relevant_documents(query, 1))
    adv_vector_tree_search = VectorDBAdvancedSearchEngine(db)
    print(adv_vector_tree_search.retrieve_relevant_documents(query, 5))
    assert result, "Result is None"


if __name__ == "__main__":
    main()
