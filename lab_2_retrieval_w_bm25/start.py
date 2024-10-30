"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf, calculate_spearman,
                                         calculate_tf, calculate_tf_idf, load_index, rank_documents,
                                         remove_stopwords, save_index, tokenize)


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
    av_doc_len = sum(len(document) for document in documents) / len(documents)
    preprocessed_documents = []
    for doc in documents:
        tokenized_doc = tokenize(doc)
#        if not isinstance(tokenized_doc, list):
#            return
        preprocessed_doc = remove_stopwords(tokenized_doc, stopwords)
#        if not isinstance(preprocessed_doc, list):
#            return
        preprocessed_documents.append(preprocessed_doc)
    vocabulary = build_vocabulary(preprocessed_documents)
    tf = []
    for document in preprocessed_documents:
#        if not isinstance(document, str):
#            return
        tf_doc = calculate_tf(vocabulary, document)
#        if not isinstance(tf_doc, list):
#            return
        tf.append(tf_doc)
    tf_idf = []
#    if not isinstance(vocabulary, list):
#        return
    idf = calculate_idf(vocabulary, preprocessed_documents)
#    if not isinstance(idf, list):
#        return
    for tf_document in tf:
        tf_idf_doc = calculate_tf_idf(tf_document, idf)
        tf_idf.append(tf_idf_doc)
    bm25 = []
    for document in preprocessed_documents:
        bm25_doc = calculate_bm25(vocabulary,document,idf,1.5,0.75,av_doc_len,len(document))
        bm25.append(bm25_doc)
    optimised_bm25 = []
    for document_ in preprocessed_documents:
        optimised_bm25_doc = calculate_bm25_with_cutoff(vocabulary, document_, idf,\
                                                        0.2, 1.5, 0.75, av_doc_len, len(document_))
        optimised_bm25.append(optimised_bm25_doc)
    query = 'Which fairy tale has Fairy Queen?'
    save_index(optimised_bm25, 'assets/metrics.json')
    optimised_bm25_metrics = load_index('assets/metrics.json')

    ranked_tf_idf = rank_documents(tf_idf, query, stopwords)
    ranked_bm25 = rank_documents(bm25, query, stopwords)
    ranked_optimised_bm25 = rank_documents(optimised_bm25_metrics, query, stopwords)
    ranks_tf_idf = [rank_tf_idf[0] for rank_tf_idf in ranked_tf_idf]
    ranks_bm25 = [rank_bm25[0] for rank_bm25 in ranked_bm25]
    ranks_optimised_bm25 = [rank_optimised_bm25[0] for rank_optimised_bm25 in ranked_optimised_bm25]
    spearman_tf_idf_and_bm25 = calculate_spearman(ranks_tf_idf,ranks_bm25)
    spearman_bm25_and_optimised_bm25 = calculate_spearman(ranks_bm25, ranks_optimised_bm25)
    spearman_tf_idf_and_optimised_bm25 = (ranks_tf_idf, ranks_optimised_bm25)

    result = ranks_optimised_bm25
    print(ranks_optimised_bm25)
    assert result, "Result is None"



if __name__ == "__main__":
    main()
