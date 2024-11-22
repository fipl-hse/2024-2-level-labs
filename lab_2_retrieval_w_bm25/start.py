"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable,
# too-many-branches, too-many-statements, duplicate-code

from lab_2_retrieval_w_bm25.main import (
    build_vocabulary,
    calculate_bm25,
    calculate_bm25_with_cutoff,
    calculate_idf,
    calculate_spearman,
    calculate_tf,
    calculate_tf_idf,
    load_index,
    rank_documents,
    remove_stopwords,
    save_index,
    tokenize,
)


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
    tokens = []
    clear_tokens = []
    for doc in documents:
        tok_doc = tokenize(doc)
        tokens.append(tok_doc)
        if tok_doc is not None:
            clear_tok_doc = remove_stopwords(tok_doc, stopwords)
            if clear_tok_doc is not None:
                clear_tokens.append(clear_tok_doc)
    unique_tokens = build_vocabulary(clear_tokens)
    if not isinstance(unique_tokens, list):
        return None
    idf = calculate_idf(unique_tokens, clear_tokens)
    metrics_tf_idf = []
    metrics_bm_25 = []
    bm_25_advanced = []
    adv_len = sum(len(tokens) for tokens in clear_tokens) / len(clear_tokens)
    for tok in tokens:
        if not isinstance(tok, list):
            return None
        tf = calculate_tf(unique_tokens, tok)
        if not isinstance(tf, dict) or not isinstance(idf, dict):
            return None
        tf_idf = calculate_tf_idf(tf, idf)
        if tf_idf is not None:
            metrics_tf_idf.append(tf_idf)
        bm_25 = calculate_bm25(unique_tokens, tok, idf, 1.5, 0.75, adv_len, len(tok))
        if bm_25 is not None:
            metrics_bm_25.append(bm_25)
        bm_25_adv = calculate_bm25_with_cutoff(
            unique_tokens, tok, idf, 0.2, 1.5, 0.75, adv_len, len(tok)
        )
        if bm_25_adv is not None:
            bm_25_advanced.append(bm_25_adv)
    rank_tf_idf = rank_documents(metrics_tf_idf, "Which fairy tale has Fairy Queen?", stopwords)
    rank_bm_25 = rank_documents(metrics_bm_25, "Which fairy tale has Fairy Queen?", stopwords)
    save_index(bm_25_advanced, "assets/metrics.json")
    loaded = load_index("assets/metrics.json")
    if not isinstance(loaded, list):
        return
    rank_from_json = rank_documents(loaded, "Which fairy tale has Fairy Queen?", stopwords)
    spearman_tf_idf = calculate_spearman(
        [num[0] for num in rank_tf_idf], [golden[0] for golden in rank_from_json]
    )
    spearman_bm_25 = calculate_spearman(
        [num[0] for num in rank_bm_25], [golden[0] for golden in rank_from_json]
    )
    spearman_bm_25_adv = calculate_spearman(
        [num[0] for num in rank_from_json], [golden[0] for golden in rank_from_json]
    )
    print(f"Golden rank: {[golden[0] for golden in rank_from_json]}")
    print(f"Spearman tf_idf: {spearman_tf_idf}")
    print(f"Spearman BM25: {spearman_bm_25}")
    print(f"Spearman BM25 with cutoff: {spearman_bm_25_adv}")
    result = spearman_bm_25_adv
    assert result, "Result is None"


if __name__ == "__main__":
    main()
