"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf,
                                         calculate_tf, calculate_tf_idf, rank_documents,
                                         remove_stopwords, tokenize)


def main() -> None:
    """
    Launches an implementation
    """
    paths_to_texts = [
        "assets/fairytale_1.txt",
        "assets/fairytale_2.txt",
        "assets/fairytale_3.txt",
        "assets/fairytale_4.txt",
        "assets/fairytale_5.txt",
        "assets/fairytale_6.txt",
        "assets/fairytale_7.txt",
        "assets/fairytale_8.txt",
        "assets/fairytale_9.txt",
        "assets/fairytale_10.txt",
    ]

    documents = []
    for path in paths_to_texts:
        with open(path, "r", encoding="utf-8") as file:
            documents.append(file.read())

    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")

    tokenized_documents = []
    cleared_documents = []
    for document in documents:
        tokenized_doc = tokenize(document)
        if tokenized_doc is not None:
            tokenized_documents.append(tokenized_doc)
            cleared_documents.append(remove_stopwords(tokenized_doc, stopwords))

    vocabulary = build_vocabulary(tokenized_documents)
    if not vocabulary:
        return None

    tf_idf_doc = []
    idf_check = calculate_idf(vocabulary, cleared_documents)
    idf = idf_check if idf_check is not None else {}
    for cleared_doc in cleared_documents:
        tf = calculate_tf(vocabulary, cleared_doc)
        if tf and isinstance(tf, dict):
            tf_idf = calculate_tf_idf(tf, idf)
            if tf_idf:
                tf_idf_doc.append(tf_idf)

    avg_len_doc = sum(len(doc) for doc in cleared_documents) / len(cleared_documents)
    bm25_doc = []
    for cleared_doc in cleared_documents:
        doc_len = len(cleared_doc)
        bm25 = calculate_bm25(vocabulary, cleared_doc, idf, 1.5, 0.75, avg_len_doc, doc_len)
        if bm25 is not None:
            bm25_doc.append(bm25)

    query = "Which fairy tale has Fairy Queen?"
    tokenized_query = tokenize(query)
    cleared_query = remove_stopwords(tokenized_query, stopwords)
    cleared_query_str = " ".join(cleared_query) if cleared_query else ""

    tf_idf_ranking = rank_documents(tf_idf_doc, cleared_query_str, stopwords)
    print(tf_idf_ranking)

    bm25_ranking = rank_documents(bm25_doc, cleared_query_str, stopwords)
    print(bm25_ranking)

    print(bm25_doc)
    print(tf_idf_doc)
    print(tokenized_documents[0])
    print(cleared_documents[0])
    print(vocabulary)

    result = cleared_documents
    assert result, "Result is None"


if __name__ == "__main__":
    main()
