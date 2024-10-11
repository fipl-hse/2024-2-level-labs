"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
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
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    tokenized_all_documents = []
    tf_idf_list = []
    bm_list = []
    bm_w_cutoff_list = []
    avg = 0.0
    k1 = 1.5
    b = 0.75
    alpha = 0.2
    query = 'Which fairy tale has Fairy Queen?'
    for document in documents:
        avg += len(document)
        tokenized_document = tokenize(document)
        if tokenized_document is not None:
            tokenized_document = remove_stopwords(tokenized_document, stopwords)
        if tokenized_document is not None:
            tokenized_all_documents.append(tokenized_document)
    avg = avg / len(documents)
    vocab = build_vocabulary(tokenized_all_documents)
    if vocab is None:
        return
    idf = calculate_idf(vocab, tokenized_all_documents)
    if idf is None:
        return
    for tokenized_document in tokenized_all_documents:
        if tokenized_document is None:
            return
        length = len(tokenized_document)
        # print(tokenized_document)
        tf_dict = calculate_tf(vocab, tokenized_document)
        if tf_dict is not None:
            tf_idf_for_doc = calculate_tf_idf(tf_dict, idf)
            if tf_idf_for_doc is not None:
                tf_idf_list.append(tf_idf_for_doc)
        bm_for_doc = calculate_bm25(vocab, tokenized_document, idf, k1, b, avg, length)
        if bm_for_doc is not None:
            bm_list.append(bm_for_doc)
        bm_w_cutoff_for_doc = calculate_bm25_with_cutoff(
            vocab, tokenized_document, idf, alpha, k1, b, avg, length)
        if bm_w_cutoff_for_doc is not None:
            bm_w_cutoff_list.append(bm_w_cutoff_for_doc)
    # print(tf_idf_list)
    # print(bm_list)
    if not iter(rank_documents(tf_idf_list, query, stopwords)):
        return
    if not iter(rank_documents(bm_list, query, stopwords)):
        return
    tf_idf_rank_tuples = rank_documents(tf_idf_list, query, stopwords)
    if not all(isinstance(i, tuple) for i in tf_idf_rank_tuples):
        return
    bm_rank_tuples = rank_documents(bm_list, query, stopwords)
    print(tf_idf_rank_tuples, bm_rank_tuples)
    bm_rank = [tup[0] for tup in bm_rank_tuples]
    tf_idf_rank = [tup[0] for tup in tf_idf_rank_tuples]
    save_index(bm_w_cutoff_list, 'assets/metrics.json')
    loaded_docs_list = load_index('assets/metrics.json')
    if loaded_docs_list is None:
        return
    if rank_documents(loaded_docs_list, query, stopwords) is None:
        return
    bm_w_cutoff_rank_tuples = rank_documents(loaded_docs_list, query, stopwords)
    bm_w_cutoff_rank = [tup[0] for tup in bm_w_cutoff_rank_tuples]
    tf_result = calculate_spearman(tf_idf_rank, bm_w_cutoff_rank)
    bm_result = calculate_spearman(bm_rank, bm_w_cutoff_rank)
    print(tf_result, bm_result)
    result = bm_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
