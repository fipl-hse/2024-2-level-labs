"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf,
                                         calculate_tf, calculate_tf_idf, rank_documents,
                                         remove_stopwords, tokenize)


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

    tokenized_documents = []
    for str_of_tokens in documents:
        list_of_tokens = tokenize(str_of_tokens)
        tokenized_documents.append(list_of_tokens)

    if documents is None or tokenized_documents is None:
        return None
    for every in documents:
        if every is None:
            return None
    for list_of_tokens in tokenized_documents:
        if list_of_tokens is None:
            return None
        for word in list_of_tokens:
            if word is None:
                return None

    with (open("assets/stopwords.txt", "r", encoding="utf-8") as file):
        stopwords = file.read().split("\n")

        if stopwords is None:
            return None

        vocab_made_of_tok_doc = build_vocabulary(tokenized_documents)
        tok_doc_without_stopwords = remove_stopwords(tokenized_documents, stopwords)
        if vocab_made_of_tok_doc is None or tok_doc_without_stopwords is None:
            return None

        for doc in tokenized_documents:
            tf_dict = calculate_tf(vocab_made_of_tok_doc,
                              tok_doc_without_stopwords)
            idf_dict = calculate_idf(vocab_made_of_tok_doc, tokenized_documents)

            if tf_dict is None or idf_dict is None:
                return None

            tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)

        if (not isinstance(tf_idf_dict, dict) or not tf_idf_dict
                or not all(isinstance(tf_idf_dict_keys, str)
                           for tf_idf_dict_keys in tf_idf_dict.keys()
                or not all(isinstance(tf_idf_dict_values, float)
                           for tf_idf_dict_values in tf_idf_dict.values()))):
            return None

        avg_doc_len_list = []
        for tokenized_doc in tokenized_documents:
            avg_doc_len_list.append(len(tokenized_doc))
        avg_doc_len = sum(avg_doc_len_list)/len(tokenized_doc)

        doc_len = len(list_of_tokens)
        bm_25 = calculate_bm25(vocab_made_of_tok_doc,
                           tokenized_documents, idf_dict,
                           1.5, 0.75, avg_doc_len, doc_len)

    rank_result = rank_documents(tf_idf_dict, 'A story about a wizard boy in a tower!', stopwords)
    rank_result = rank_documents(bm_25, 'A story about a wizard boy in a tower!', stopwords)

    result = rank_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
