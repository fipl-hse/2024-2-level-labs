"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf, calculate_tf,
                                         calculate_tf_idf, load_index, rank_documents,
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
    tokenized_all_documents = []
    avg = 0.0
    tf_idf_list = []
    k1 = 1.5
    b = 0.75
    alpha = 0.2
    bm_list = []
    bm_w_cutoff_list = []
    query = 'Which fairy tale has Fairy Queen?'
    for document in documents:
        avg += len(document)
        tokenized_all_documents.append(remove_stopwords(tokenize(document), stopwords))
    avg = avg / len(documents)
    vocab = build_vocabulary(tokenized_all_documents)
    for document in documents:
        length = len(document)
        tokenization_res = remove_stopwords(tokenize(document), stopwords)
        print(tokenization_res)
        idf = calculate_idf(vocab, tokenized_all_documents)
        tf_idf_list.append(calculate_tf_idf(calculate_tf(vocab, tokenization_res), idf))
        bm_list.append(calculate_bm25(vocab, tokenization_res, idf, k1, b, avg, length))
        bm_w_cutoff_list.append(calculate_bm25_with_cutoff(vocab, tokenization_res, idf, alpha, k1, b, avg, length))
    print(tf_idf_list)
    print(bm_list)
    print(rank_documents(tf_idf_list, query, stopwords))
    print(rank_documents(bm_list, query, stopwords))
    save_index(bm_w_cutoff_list, 'assets/metrics.json')
    loaded_docs_list = load_index('assets/metrics.json')
    result = rank_documents(loaded_docs_list, query, stopwords)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
