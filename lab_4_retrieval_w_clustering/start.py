"""
Laboratory Work #4 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable

import json

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
    documents = ''.join(open_files()[0]) #since we need str
    stopwords = open_files()[1]
    paragraphs = get_paragraphs(documents)
    query = 'Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления'
    dbase = DocumentVectorDB(stopwords)
    dbase.put_corpus(paragraphs)
    searchengine = VectorDBSearchEngine(dbase)
    print(searchengine.retrieve_relevant_documents(query,3))

    cluster_searchengine = ClusteringSearchEngine(dbase)
    # print(cluster_searchengine.retrieve_relevant_documents(query, 5))
    cluster_searchengine.make_report(5, 'assets/report.json')
    with open('assets/report.json', 'r', encoding='utf-8') as file:
        state = json.load(file)
    print(state)

    for i in range(1,15):
        new_cluster_searchengine = ClusteringSearchEngine(dbase,n_clusters=i)
        print(new_cluster_searchengine.calculate_square_sum())

    tree_engine = VectorDBTreeSearchEngine(dbase)
    print(tree_engine.retrieve_relevant_documents(query,1))

    advanced_engine = VectorDBAdvancedSearchEngine(dbase)
    print(advanced_engine.retrieve_relevant_documents(query,5))

    result = '???'
    assert result, "Result is None"



if __name__ == "__main__":
    main()
