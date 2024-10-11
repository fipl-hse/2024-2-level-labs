"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf,
                                         calculate_tf, calculate_tf_idf, remove_stopwords, tokenize)


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
        i = 0
        tokenized_documents = []
        for each_read_text in documents:
            list_of_tokens = tokenize(documents[i])
            i += 1
            tokenized_documents.append(list_of_tokens)

    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")

        for doc in tokenized_documents:
            tf_dict = calculate_tf(build_vocabulary(tokenized_documents),
                              remove_stopwords(tokenized_documents, stopwords))
            idf_dict = calculate_idf(build_vocabulary(tokenized_documents), tokenized_documents)
            tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)

            '''avg_doc_len = len
            for list_of_tokens in documents:
                doc_len = len(list_of_tokens)
                bm_25 = calculate_bm25(build_vocabulary(tokenized_documents),
                                   tokenized_documents, idf_dict,
                                   1.5, 0.75, avg_doc_len, doc_len)'''

    if (not isinstance(tf_idf_dict, dict) or not tf_idf_dict
            or not all(isinstance(tf_idf_dict_keys, str) for tf_idf_dict_keys in tf_idf_dict.keys()
            or not all(isinstance(tf_idf_dict_values, float)
                       for tf_idf_dict_values in tf_idf_dict.values()))):
        return

    result = tf_idf_dict
    assert result, "Result is None"


if __name__ == "__main__":
    main()
