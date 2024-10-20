"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf,
                                         calculate_spearman, calculate_tf,
                                         calculate_tf_idf, load_index,
                                         rank_documents, remove_stopwords, save_index,
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

    query = "A story about a wizard boy in a tower!"

    result = None
    tokenized_documents = []
    for document in documents:
        tokens: list[str] = tokenize(document)
        tokenized_documents.append(tokens)

    tokenized_documents_without_stopwords = []
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
        for tokenized_doc in tokenized_documents:
            without_stopwords = remove_stopwords(tokenized_doc, stopwords)
            if not without_stopwords:
                return None
            tokenized_documents_without_stopwords.append(without_stopwords)

    vocab = build_vocabulary(tokenized_documents)
    if not vocab:
        return None

    list_for_tf_idf = []
    for tokenized_doc_without in tokenized_documents_without_stopwords:
        tf = calculate_tf(vocab, tokenized_doc_without)
        idf = calculate_idf(vocab, tokenized_documents_without_stopwords)
        if ((not calculate_tf(vocab, tokenized_doc_without) or
                not calculate_idf(vocab, tokenized_documents_without_stopwords)) or
                not isinstance(tf, dict) or not isinstance(idf, dict)):
            return None
        tf_idf = calculate_tf_idf(tf, idf)
        if not tf_idf:
            return None
        list_for_tf_idf.append(tf_idf)

    avg_len = []
    for doc in tokenized_documents_without_stopwords:
        avg_len.append(len(doc))
    avg_len_doc = sum(avg_len) / len(tokenized_documents_without_stopwords)

    list_for_bm25 = []
    for doc in tokenized_documents_without_stopwords:
        doc_len = len(doc)
        if idf is None:
            return None
        bm25 = calculate_bm25(vocab, doc, idf, 1.5, 0.75, avg_len_doc, doc_len)
        list_for_bm25.append(bm25)

    list_for_bm25_without_cutoff = []
    for doc_bm25 in tokenized_documents_without_stopwords:
        doc_len = len(doc_bm25)
        if idf is None:
            return None
        bm25_score = calculate_bm25_with_cutoff(vocab, doc_bm25, idf,
                                                0.2, 1.5, 0.75, avg_len_doc, doc_len)
        list_for_bm25_without_cutoff.append(bm25_score)

    rank = rank_documents(list_for_tf_idf, query, stopwords)
    rank = rank_documents(list_for_bm25, query, stopwords)

    save_index(list_for_bm25_without_cutoff, 'assets/metrics.json')
    load_docs = load_index('assets/metrics.json')
    if load_docs is None:
        return None
    cutoff_tuples = rank_documents(load_docs, query, stopwords)
    if cutoff_tuples is None:
        return None


    assert result, "Result is None"



if __name__ == "__main__":
    main()
