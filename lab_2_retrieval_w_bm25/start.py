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
    clean_documents = []
    for document in documents:
        tokenized_document = tokenize(document)
        if tokenized_document:
            clean_document = remove_stopwords(tokenized_document, stopwords)
            if clean_document:
                clean_documents.append(clean_document)
    vocab = build_vocabulary(clean_documents)
    if not vocab:
        return None
    idf = calculate_idf(vocab, clean_documents)
    if not idf:
        return None
    avg_doc_len = sum(len(document) for document in clean_documents) / len(clean_documents)
    tf_idf_list = []
    bm25_list = []
    for doc in clean_documents:
        tf = calculate_tf(vocab, doc)
        if not tf:
            return None
        tf_idf = calculate_tf_idf(tf, idf)
        if not tf_idf:
            return None
        tf_idf_list.append(tf_idf)
        bm25 = calculate_bm25(vocab, doc, idf, 1.5, 0.75, avg_doc_len, len(doc))
        if not bm25:
            return None
        bm25_list.append(bm25)
    query = "Which fairy tale has Fairy Queen?"
    tfidf_ranking = rank_documents(tf_idf_list, query, stopwords)
    bm25_ranking = rank_documents(bm25_list, query, stopwords)
    if not tfidf_ranking or not bm25_ranking:
        return None
    print(f"{tfidf_ranking}\n{bm25_ranking}")
    result = [index_score_1 for index_score_1, index_score_2 in zip(tfidf_ranking, bm25_ranking)
              if index_score_1 == index_score_2]
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
