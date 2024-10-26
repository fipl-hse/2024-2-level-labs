"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_idf, calculate_tf,
                                         calculate_tf_idf, remove_stopwords, tokenize)

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


    # for grade 6:

    # metrics_dict = list[dict[str:bool]]
    # metrics_dict = [{word1: TF-IDF1, word2: TF-IDF2...}, ...]
    metrics = []

    for doc in documents:
        tokenized_text = tokenize(doc)
        clean_text = remove_stopwords(tokenized_text, stopwords)
        vocab = build_vocabulary(clean_text)
        tf_metric = calculate_tf(vocab, clean_text)
        idf_metric = calculate_idf(vocab, clean_text)
        tf_idf_metric = calculate_tf_idf(tf_metric, idf_metric)
        metrics.append(tf_idf_metric)


    print(metrics)

    result = metrics
    assert result, "Result is None"


if __name__ == "__main__":
    main()
