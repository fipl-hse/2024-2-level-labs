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
    tokenized_documents = [remove_stopwords(tokenize(document), stopwords) for document in documents]
    vocab = build_vocabulary(tokenized_documents)
    tf_list = [calculate_tf(vocab, document) for document in tokenized_documents]
    idf = calculate_idf(vocab, tokenized_documents)
    tf_idf_list = [calculate_tf_idf(tf, idf) for tf in tf_list]
    print(tf_idf_list)


if __name__ == "__main__":
    main()
