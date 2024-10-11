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
    clean_documents = []
    for document in documents:
        tokenized_document = tokenize(document)
        if tokenized_document:
            clean_document = remove_stopwords(tokenized_document, stopwords)
            if clean_document:
                clean_documents.append(clean_document)
    vocab = build_vocabulary(clean_documents)
    if not vocab:
        return
    idf = calculate_idf(vocab, clean_documents)
    for clean_document in clean_documents:
        tf = calculate_tf(vocab, clean_document)
        if not tf or not idf:
            return
        tf_idf_dict = calculate_tf_idf(tf, idf)
        tf_idf_list = []
        if tf_idf_dict:
            tf_idf_list.append(tf_idf_dict)
        result = tf_idf_list
        print(result)
        assert result, "Result is None"


if __name__ == "__main__":
    main()
