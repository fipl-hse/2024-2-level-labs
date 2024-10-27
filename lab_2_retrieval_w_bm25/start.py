"""
Laboratory Work #2 starter
"""
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_idf, calculate_tf,
                                         calculate_tf_idf, remove_stopwords, tokenize)

# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code


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
    result = remove_stopwords
    assert result, "Result is None"
    tokenized_docs = []
    for doc in documents:
        tokenized_doc = tokenize(doc)
        tokenized_docs.append(tokenized_doc)
    # print(tokenized_docs)
    meaningful_docs = []
    for doc in tokenized_docs:
        meaningful_doc = remove_stopwords(doc, stopwords)
        meaningful_docs.append(meaningful_doc)
    # print(meaningful_docs)
    vocab = build_vocabulary(meaningful_docs)
    # print (vocab)
    tf_values_lst = []
    for doc in meaningful_docs:
        tf_doc = calculate_tf(vocab, doc)
        tf_values_lst.append(tf_doc)
    # print(tf_values_lst)
    for doc in tf_values_lst:
        print(calculate_tf_idf(doc, calculate_idf(vocab, meaningful_docs)))


if __name__ == "__main__":
    main()
