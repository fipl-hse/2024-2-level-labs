"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (tokenize, remove_stopwords, build_vocabulary,
                                         calculate_tf, calculate_idf, calculate_tf_idf,
                                         calculate_bm25, rank_documents, calculate_bm25_with_cutoff,
                                         save_index, load_index, calculate_spearman)


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

    avg_doc_len = 0.0
    clear_documents = []
    for document in documents:
        avg_doc_len += len(document)
        tokenized_document = tokenize(document)
        if not tokenized_document:
            return None
        clear_document = remove_stopwords(tokenized_document, stopwords)
        if not clear_document:
            return None
        clear_documents.append(clear_document)
    vocab = build_vocabulary(clear_documents)
    if not vocab:
        return None
    idf_dict = calculate_idf(vocab, clear_documents)
    if not idf_dict:
        return None
    avg_doc_len /= len(clear_documents)
    alpha = 0.2
    k1 = 1.5
    b = 0.75
    tf_idf_list = []
    bm25_list = []
    bm25_plus_list = []

    for document in clear_documents:
        if not document:
            return None
        tf_dict = calculate_tf(vocab, document)
        if not tf_dict:
            return None
        doc_len = len(document)
        tf_idf = calculate_tf_idf(tf_dict, idf_dict)
        bm25 = calculate_bm25(vocab, document, idf_dict, k1, b, avg_doc_len, doc_len)
        optimized_bm25 = calculate_bm25_with_cutoff(vocab, document, idf_dict,
                                                    alpha, k1, b, avg_doc_len, doc_len)
        if not tf_idf or not bm25 or not optimized_bm25:
            return None
        tf_idf_list.append(tf_idf)
        bm25_list.append(bm25)
        bm25_plus_list.append(optimized_bm25)

    query = 'Which fairy tale has Fairy Queen?'
    ranked_tf_idf = rank_documents(tf_idf_list, query, stopwords)
    ranked_bm25 = rank_documents(bm25_list, query, stopwords)
    ranked_optimized_bm25 = rank_documents(bm25_plus_list, query, stopwords)
    if not ranked_tf_idf or not ranked_bm25 or not ranked_optimized_bm25:
        return None

    save_index(bm25_plus_list, 'assets/metrics.json')
    loaded_index = load_index('assets/metrics.json')
    if not loaded_index:
        return None
    ranked_index = rank_documents(loaded_index, query, stopwords)
    if not ranked_index:
        return None

    golden_rank = list(list(zip(*ranked_optimized_bm25))[0])
    tf_idf_spearman = calculate_spearman(list(list(zip(*ranked_tf_idf))[0]), golden_rank)
    bm25_spearman = calculate_spearman(list(list(zip(*ranked_bm25))[0]), golden_rank)
    if not tf_idf_spearman or not bm25_spearman:
        return None

    result = (tf_idf_spearman, bm25_spearman)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
