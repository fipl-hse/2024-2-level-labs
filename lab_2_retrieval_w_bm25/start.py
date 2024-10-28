"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

from lab_2_retrieval_w_bm25.main import (tokenize, remove_stopwords, build_vocabulary,
                                         calculate_tf, calculate_idf, calculate_tf_idf)


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

    list_of_lists = []

    for text in documents:
        list_of_lists.append(remove_stopwords(tokenize(text), stopwords))

    doc_vocab = build_vocabulary(list_of_lists)
    tf_idf_list = []

    for i in range(len(list_of_lists)):
        tf_idf_list.append(calculate_tf_idf(calculate_tf(doc_vocab, list_of_lists[i]),
                           calculate_idf(doc_vocab, list_of_lists)))

    result = tf_idf_list
    assert result, "Result is None"

    print(result)


if __name__ == "__main__":
    main()

