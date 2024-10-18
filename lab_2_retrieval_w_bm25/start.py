"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable,
# too-many-branches, too-many-statements, duplicate-code
from main import (build_vocabulary, calculate_tf_idf, remove_stopwords, tokenize,
                  calculate_bm25, calculate_bm25_with_cutoff, calculate_idf, calculate_spearman,
                  calculate_tf, load_index, rank_documents, save_index)


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
    clear_tokens = [remove_stopwords(tokenize(doc), stopwords) for doc in documents]
    unique_tokens = build_vocabulary(clear_tokens)
    idf = calculate_idf(unique_tokens, clear_tokens)
    metrics_tf_idf = []
    metrics_bm_25 = []
    bm_25_advanced = []
    adv_len = sum(len(tokens) for tokens in clear_tokens)/len(clear_tokens)
    for tokens in clear_tokens:
        tf = calculate_tf(unique_tokens, tokens)
        metrics_tf_idf.append(calculate_tf_idf(tf, idf))
        metrics_bm_25.append(calculate_bm25(unique_tokens, tokens, idf,
                                            1.5, 0.75, adv_len, len(tokens)))
        bm_25_advanced.append(calculate_bm25_with_cutoff(unique_tokens, tokens, idf,
                              0.2, 1.5, 0.75, adv_len, len(tokens)))
    rank_tf_idf = rank_documents(metrics_tf_idf,
                                 'Which fairy tale has Fairy Queen?',
                                 stopwords)
    rank_bm_25 = rank_documents(metrics_bm_25,
                                'Which fairy tale has Fairy Queen?',
                                stopwords)
    save_index(bm_25_advanced, "assets/metrics.json")
    rank_from_json = rank_documents(load_index("assets/metrics.json"),
                                    'Which fairy tale has Fairy Queen?',
                                    stopwords)
    spearman_tf_idf = calculate_spearman([num[-1] for num in rank_tf_idf],
                                         [golden[-1] for golden in rank_from_json])
    print(spearman_tf_idf)

    result = None
    return None
    # assert result, "Result is None"


if __name__ == "__main__":
    main()
