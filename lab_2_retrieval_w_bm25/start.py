"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf,
                                         calculate_spearman, calculate_tf, calculate_tf_idf,
                                         load_index, rank_documents, remove_stopwords, save_index,
                                         tokenize)


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

    query = 'Which fairy tale has Fairy Queen?'

    tokenized_documents = []
    tokenized_documents_without_stopwords = []
    list_for_tf_idf = []
    list_for_bm25 = []
    list_for_bm25_without_cutoff = []
    k1 = 1.5
    b = 0.75
    alpha = 0.2

    for document in documents:
        tokens: list[str] = tokenize(document) or []
        tokenized_documents.append(tokens)

    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
        for tokenized_doc in tokenized_documents:
            without_stopwords = remove_stopwords(tokenized_doc, stopwords)
            if without_stopwords:
                tokenized_documents_without_stopwords.append(without_stopwords)

    vocab = build_vocabulary(tokenized_documents_without_stopwords)
    if not vocab:
        return
    idf_check = calculate_idf(vocab, tokenized_documents_without_stopwords)
    idf: dict[str, float] = idf_check if idf_check is not None else {}

    for tokenized_doc_without in tokenized_documents_without_stopwords:
        tf = calculate_tf(vocab, tokenized_doc_without)
        if tf and idf and isinstance(tf, dict) and isinstance(idf, dict):
            tf_idf = calculate_tf_idf(tf, idf)
            if tf_idf:
                list_for_tf_idf.append(tf_idf)

    avg_len = [len(doc) for doc in tokenized_documents_without_stopwords]
    avg_len_doc = sum(avg_len) / len(tokenized_documents_without_stopwords) \
        if avg_len else 0

    for doc in tokenized_documents_without_stopwords:
        doc_len = len(doc)
        bm25 = calculate_bm25(vocab, doc, idf, k1, b, avg_len_doc, doc_len)
        if bm25 is not None:
            list_for_bm25.append(bm25)

    for doc_bm25 in tokenized_documents_without_stopwords:
        doc_len = len(doc_bm25)
        bm25_score = calculate_bm25_with_cutoff(vocab, doc_bm25, idf,
                                                alpha, k1, b, avg_len_doc, doc_len)
        if bm25_score is not None:
            list_for_bm25_without_cutoff.append(bm25_score)

    print(list_for_bm25_without_cutoff)

    save_index(list_for_bm25_without_cutoff, 'assets/metrics.json')
    load_docs = load_index('assets/metrics.json')
    if load_docs is None:
        return

    rank_for_tf_idf = rank_documents(list_for_tf_idf, query, stopwords)
    rank_for_bm = rank_documents(list_for_bm25, query, stopwords)
    cutoff_tuples = rank_documents(load_docs, query, stopwords)

    rank_tf_idf = [number[0] for number in rank_for_tf_idf]
    rank_bm = [number[0] for number in rank_for_bm]
    rank_bm_without = [number[0] for number in cutoff_tuples]

    result_tf = calculate_spearman(rank_tf_idf, rank_bm_without)
    result_bm = calculate_spearman(rank_bm, rank_bm_without)
    print(rank_bm_without)
    print(result_tf, result_bm)

    result = result_bm

    assert result, "Result is None"


if __name__ == "__main__":
    main()
