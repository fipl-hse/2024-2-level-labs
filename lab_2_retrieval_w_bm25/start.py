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
    tf_idf = []
    for path in paths_to_texts:
        with open(path, "r", encoding="utf-8") as file:
            documents.append(file.read())
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    tokenized_documents = []
    for document in documents:
        tokenized_doc = tokenize(document)
        if tokenized_doc is not None:
            tokenized_doc = remove_stopwords(tokenized_doc, stopwords)
        if tokenized_doc is not None:
            tokenized_documents.append(tokenized_doc)
    vocab = build_vocabulary(tokenized_documents)
    if vocab is None:
        return
    idf = calculate_idf(vocab, tokenized_documents)
    if idf is None:
        return
    for tokenized_doc in tokenized_documents:
        if tokenized_doc is None:
            return
        tf = calculate_tf(vocab, tokenized_doc)
        if tf is not None:
            tf_idf_doc = calculate_tf_idf(tf, idf)
            if tf_idf_doc is not None:
                tf_idf.append(tf_idf_doc)
    if tf_idf:
        result = tf_idf
    else:
        result = None
    assert result, "Result is None"


if __name__ == "__main__":
    main()
