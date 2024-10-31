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
    result = None
    clean_documents = []
    for document in documents:
        tokenized_document = tokenize(document)
        if not tokenized_document:
            result = None
            assert result, "Result is None"
        clean_document = remove_stopwords(tokenized_document, stopwords)
        if not clean_document:
            result = None
            assert result, "Result is None"
        clean_documents.append(clean_document)
    vocab = build_vocabulary(clean_documents)
    if not vocab:
        result = None
        assert result, "Result is None"
    idf = calculate_idf(vocab, clean_documents)
    if not idf:
        result = None
        assert result, "Result is None"
    avg_doc_len = sum(len(document) for document in clean_documents) / len(clean_documents)
    tf_idf_list = []
    bm25_list = []
    for doc in clean_documents:
        tf = calculate_tf(vocab, doc)
        if not tf:
            result = None
            assert result, "Result is None"
        tf_idf = calculate_tf_idf(tf, idf)
        if not tf_idf:
            result = None
            assert result, "Result is None"
        tf_idf_list.append(tf_idf)
        bm25 = calculate_bm25(vocab, doc, idf, 1.5, 0.75, avg_doc_len, len(doc))
        if not bm25:
            result = None
            assert result, "Result is None"
        bm25_list.append(bm25)
    query = "Which fairy tale has Fairy Queen?"
    tfidf_ranking = rank_documents(tf_idf_list, query, stopwords)
    bm25_ranking = rank_documents(bm25_list, query, stopwords)
    if tfidf_ranking and bm25_ranking:
        result = list(zip(tfidf_ranking, bm25_ranking))
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
