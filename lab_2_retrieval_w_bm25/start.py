"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches,
# too-many-statements, duplicate-code
import lab_2_retrieval_w_bm25.main as m


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
        clear_document = m.remove_stopwords(m.tokenize(document), stopwords)
        clear_documents.append(clear_document)
    vocab = m.build_vocabulary(clear_documents)
    idf_dict = m.calculate_idf(vocab, clear_documents)
    avg_doc_len /= len(clear_documents)
    alpha = 0.2
    k1 = 1.5
    b = 0.75
    tf_idf_list = []
    bm25_list = []
    bm25_plus_list = []

    for document in clear_documents:
        doc_len = len(document)
        tf_idf = m.calculate_tf_idf(m.calculate_tf(vocab, document), idf_dict)
        bm25 = m.calculate_bm25(vocab, document, idf_dict, k1, b, avg_doc_len, doc_len)
        optimized_bm25 = m.calculate_bm25_with_cutoff(vocab, document, idf_dict,
                                                      alpha, k1, b, avg_doc_len, doc_len)
        tf_idf_list.append(tf_idf)
        bm25_list.append(bm25)
        bm25_plus_list.append(optimized_bm25)

    query = 'Which fairy tale has Fairy Queen?'
    ranked_tf_idf = m.rank_documents(tf_idf_list, query, stopwords)
    ranked_bm25 = m.rank_documents(bm25_list, query, stopwords)

    m.save_index(bm25_plus_list, 'assets/metrics.json')
    loaded_index = m.load_index('assets/metrics.json')
    ranked_index = m.rank_documents(loaded_index, query, stopwords)

    golden_rank = list(list(zip(*ranked_index))[0])
    tf_idf_spearman = m.calculate_spearman(list(list(zip(*ranked_tf_idf))[0]), golden_rank)
    bm25_spearman = m.calculate_spearman(list(list(zip(*ranked_bm25))[0]), golden_rank)

    result = (tf_idf_spearman, bm25_spearman)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
