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

    tokenized_document = []
    for doc in documents:
        tokenized_doc = tokenize(doc)
        if tokenized_doc is None:
            continue
        tokens_without_stopwords = remove_stopwords(tokenized_doc, stopwords)
        if tokens_without_stopwords is None:
            return
        tokenized_document.append(tokens_without_stopwords)
    #print(f'tokenized document: {tokenized_document}')
    if tokenized_document is None:
        return
    vocabulary = build_vocabulary(tokenized_document)
    if vocabulary is None:
        return
    #print(f'vocabulary with unique words: {vocabulary}')

    list_of_tf_idf = []
    idf = calculate_idf(vocabulary, tokenized_document)
    if idf is None:
        return
    for document in tokenized_document:
        tf = calculate_tf(vocabulary, document)
        if tf is None:
            return
        tf_idf = calculate_tf_idf(tf, idf)
        if tf_idf is None:
            return
        list_of_tf_idf.append(tf_idf)
    #print(f'tf idf result: {list_of_tf_idf}')

    avg_doc_len_list = []
    for i in tokenized_document:
        avg_doc_len_list.append(len(i))
    avg_doc_len = sum(avg_doc_len_list) / len(tokenized_document)
    # print(f'avg document length: {avg_doc_len}')

    list_of_bm25 = []
    list_of_bm25_with_cutoff = []
    for document in tokenized_document:
        doc_len = len(document)
        bm25 = calculate_bm25(vocabulary, document, idf, 1.5, 0.75, avg_doc_len, doc_len)
        if isinstance(bm25, dict):
            list_of_bm25.append(bm25)

        bm25_with_cutoff = calculate_bm25_with_cutoff(vocabulary, document, idf,
                                                      0.2, 1.5, 0.75, avg_doc_len, doc_len)
        if isinstance(bm25_with_cutoff, dict):
            list_of_bm25_with_cutoff.append(bm25_with_cutoff)

    # print(f'bm25 result: {list_of_bm25}')
    # print(f'bm25 with cutoff result: {list_of_bm25_with_cutoff}')

    query = 'Which fairy tale has Fairy Queen?'
    #query = 'A story about a wizard boy in a tower!'
    tf_idf_ranking = rank_documents(list_of_tf_idf, query, stopwords)
    bm25_ranking = rank_documents(list_of_bm25, query, stopwords)
    bm25_with_cutoff_ranking = rank_documents(list_of_bm25_with_cutoff, query, stopwords)

    file_path = 'assets/metrics.json'
    save_index(list_of_bm25_with_cutoff, file_path)
    indexes = load_index(file_path)
    if indexes is None:
        return

    r_tfidf = [elem[0] for elem in tf_idf_ranking]
    r_bm25 = [elem[0] for elem in bm25_ranking]
    r_bm25_with_cutoff = [elem[0] for elem in bm25_with_cutoff_ranking]

    tfidf_spearman_corr = calculate_spearman(r_tfidf, r_bm25_with_cutoff)
    bm25_spearman_corr = calculate_spearman(r_bm25, r_bm25_with_cutoff)
    print(f'tfidf spearman: {tfidf_spearman_corr}')
    print(f'bm25 spearman: {bm25_spearman_corr}')

    result = bm25_spearman_corr, tfidf_spearman_corr
    assert result, "Result is None"


if __name__ == "__main__":
    main()
