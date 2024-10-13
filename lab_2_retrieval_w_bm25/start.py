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

    documents_preprocessed = []
    for document in documents:
        document_tokenized = tokenize(document)
        if document_tokenized is None:
            return None
        document_preprocessed = remove_stopwords(document_tokenized, stopwords)
        if document_preprocessed is None:
            return None
        documents_preprocessed.append(document_preprocessed)

    vocabulary: list[str] = build_vocabulary(documents_preprocessed)

    tf_documents: list[dict[str, float]] = []
    for document in documents_preprocessed:
        if not isinstance(document, list):
            return None
        tf = calculate_tf(vocabulary, document)
        if tf is None:
            return None
        tf_documents.append(tf)

    idf_documents = calculate_idf(vocabulary, documents_preprocessed)
    tf_idf_documents: list[dict[str, float]] = []
    for tf in tf_documents:
        if not isinstance(idf_documents, dict) \
                or not all(isinstance(key, str) for key in idf_documents) \
                or not all(isinstance(value, float) for value in idf_documents.values()):
            return None
        tf_idf = calculate_tf_idf(tf, idf_documents)
        if tf_idf is None:
            return None
        tf_idf_documents.append(tf_idf)

    bm25_documents: list[dict[str, float]] = []
    avg_doc_len = sum(len(document) for document in documents) / len(documents)
    for document in documents_preprocessed:
        if not isinstance(document, list) or not all(isinstance(item, str) for item in document):
            return None
        bm25 = calculate_bm25(vocabulary, document, idf_documents,
                              avg_doc_len=avg_doc_len, doc_len=len(document))
        if bm25 is None:
            return None
        bm25_documents.append(bm25)

    query = 'Which fairy tale has Fairy Queen?'
    tf_idf_ranked = rank_documents(tf_idf_documents, query, stopwords)
    if tf_idf_ranked is None:
        return None
    bm25_ranked = rank_documents(bm25_documents, query, stopwords)
    if bm25_ranked is None:
        return None

    bm25_with_cutoff = []
    for document in documents_preprocessed:
        if not isinstance(document, list) or not all(isinstance(item, str) for item in document):
            return None
        result = calculate_bm25_with_cutoff(vocabulary, document, idf_documents, 0.2,
                                            avg_doc_len=avg_doc_len, doc_len=len(document))
        if result is None:
            return None
        bm25_with_cutoff.append(result)

    save_index(bm25_with_cutoff, 'assets/metrics.json')
    loaded_bm25_with_cutoff = load_index('assets/metrics.json')

    if not isinstance(loaded_bm25_with_cutoff, list):
        return None
    bm25_cutoff_ranked = rank_documents(loaded_bm25_with_cutoff,
                                        'Which fairy tale has Fairy Queen?', stopwords)

    tf_idf_ranks_only = [item[0] for item in tf_idf_ranked]
    bm25_cutoff_ranks_only = [item[0] for item in bm25_cutoff_ranked]
    bm25_ranks_only = [item[0] for item in bm25_ranked]

    spearman_tf_idf_bm25 = calculate_spearman(tf_idf_ranks_only, bm25_ranks_only)
    spearman_tf_idf_bm25_cutoff = calculate_spearman(tf_idf_ranks_only, bm25_cutoff_ranks_only)
    spearman_bm25_bm25_cutoff = calculate_spearman(bm25_cutoff_ranks_only, bm25_ranks_only)
    print('spearman for tf-idf and bm25:', spearman_tf_idf_bm25)
    print('spearman for tf-idf and bm25 with cutoff:', spearman_tf_idf_bm25_cutoff)
    print('spearman for bm25 and bm25 with cutoff:', spearman_bm25_bm25_cutoff)

    result = bm25_cutoff_ranks_only
    print('golden standard:', result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
