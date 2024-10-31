"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf,
                                         calculate_spearman, calculate_tf, calculate_tf_idf,
                                         load_index, rank_documents, remove_stopwords,
                                         save_index, tokenize)


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
    for doc in documents:
        tokenized_document = tokenize(doc)
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
    tf_idf_collection = []
    bm25_collection = []
    bm25_w_cutoff_collection = []
    for clean_doc in clean_documents:
        tf = calculate_tf(vocab, clean_doc)
        if not tf:
            result = None
            assert result, "Result is None"
        tf_idf = calculate_tf_idf(tf, idf)
        if not tf_idf:
            result = None
            assert result, "Result is None"
        tf_idf_collection.append(tf_idf)
        doc_len = len(clean_doc)
        bm25 = calculate_bm25(vocab, clean_doc, idf, 1.5, 0.75, avg_doc_len, doc_len)
        if not bm25:
            result = None
            assert result, "Result is None"
        bm25_collection.append(bm25)
        bm25_w_cutoff = calculate_bm25_with_cutoff(
            vocab, clean_doc, idf, 0.2, 1.5, 0.75, avg_doc_len, doc_len)
        if not bm25_w_cutoff:
            result = None
            assert result, "Result is None"
        bm25_w_cutoff_collection.append(bm25_w_cutoff)

    query = "Which fairy tale has Fairy Queen?"
    tf_idf_ranking = rank_documents(tf_idf_collection, query, stopwords)
    bm25_ranking = rank_documents(bm25_collection, query, stopwords)
    bm25_w_cutoff_ranking = rank_documents(bm25_w_cutoff_collection, query, stopwords)
    if not tf_idf_ranking or not bm25_ranking or not bm25_w_cutoff_ranking:
        result = None
        assert result, "Result is None"
    save_index(bm25_w_cutoff_collection, "assets/metrics.json")
    print(f"Loaded BM25 with cutoff: {load_index('assets/metrics.json')}")
    golden_rank = [i for i, score in bm25_w_cutoff_ranking]
    spearman_tf_idf = calculate_spearman([i for i, score in tf_idf_ranking], golden_rank)
    spearman_bm25 = calculate_spearman([i for i, score in bm25_ranking], golden_rank)
    spearman_golden = calculate_spearman(golden_rank, golden_rank)
    result = [spearman_tf_idf, spearman_bm25, spearman_golden]
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
