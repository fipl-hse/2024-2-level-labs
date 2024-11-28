"""
Laboratory Work #4 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable

from lab_4_retrieval_w_clustering.main import (
    BM25Vectorizer,
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
    return documents, stopwords


def main() -> None:
    """
    Launch an implementation.
    """
    documents, stopwords = open_files()
    document_paragraphs = []
    for document in documents:
        paragraphs = get_paragraphs(document)
        document_paragraphs.append(paragraphs)

    query = "Первый был не кто иной, как Михаил Александрович Берлиоз, председатель правления"
    n_neighbours = 5

    vectorizer = BM25Vectorizer()
    vectorizer.set_tokenized_corpus(document_paragraphs)
    vectorizer.build()
    document_vector = vectorizer.vectorize(document_paragraphs[0])
    print(document_vector)
    print()

    database = DocumentVectorDB(stopwords)
    database.put_corpus(documents)

    database_searcher = VectorDBSearchEngine(database)
    db_relevant_documents = database_searcher.retrieve_relevant_documents(query, n_neighbours)
    for db_relevant_document in db_relevant_documents:
        print(db_relevant_document)
    print()

    for cluster_number in range(1, 15):
        clustering_searcher = ClusteringSearchEngine(database, cluster_number)
        print(f"For {cluster_number} cluster(s) the error is "
              f"{clustering_searcher.calculate_square_sum()}")
        cl_relevant_documents = clustering_searcher.retrieve_relevant_documents(query, n_neighbours)
        for cl_relevant_document in cl_relevant_documents:
            print(cl_relevant_document)
        print()

    basic_searcher = VectorDBTreeSearchEngine(database)
    basic_relevant_documents = basic_searcher.retrieve_relevant_documents(query, n_neighbours)
    for basic_relevant_document in basic_relevant_documents:
        print(basic_relevant_document)
    print()

    advanced_searcher = VectorDBAdvancedSearchEngine(database)
    adv_relevant_documents = advanced_searcher.retrieve_relevant_documents(query, n_neighbours)
    for adv_relevant_document in adv_relevant_documents:
        print(adv_relevant_document)
    print()

    result = "more_relevant_documents"
    assert result, "Result is None"


if __name__ == "__main__":
    main()
