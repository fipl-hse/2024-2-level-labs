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
        if list_of_tokens is None:
            return None
        tokenized_documents.append(list_of_tokens)

    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")

        if stopwords is None or documents is None or tokenized_documents is None:
            return None

        vocab_made_of_tok_doc = build_vocabulary(tokenized_documents)
        tok_doc_without_stopwords = []
        for doc in tokenized_documents:
            without_stopwords = remove_stopwords(doc, stopwords)
            if without_stopwords is None:
                return None
            tok_doc_without_stopwords.append(without_stopwords)

        if vocab_made_of_tok_doc is None or tok_doc_without_stopwords is None:
            return None

        list_of_tf_idf_dict = []
        for doc in tok_doc_without_stopwords:
            tf_dict = calculate_tf(vocab_made_of_tok_doc, doc)
            idf_dict = calculate_idf(vocab_made_of_tok_doc, tok_doc_without_stopwords)
            if tf_dict is None or idf_dict is None:
                return None
            tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
            if tf_idf_dict is None:
                return None
            list_of_tf_idf_dict.append(tf_idf_dict)

        '''if list_of_tf_idf_dict is None or idf_dict is None:
            return None'''

        avg_doc_len_list = []
        for tok_doc in tok_doc_without_stopwords:
            avg_doc_len_list.append(len(doc))
        avg_doc_len = sum(avg_doc_len_list)/len(tok_doc_without_stopwords)

        list_of_dict_with_bm25 = []
        for doc in tok_doc_without_stopwords:
            doc_len = len(doc)
            bm_25 = calculate_bm25(vocab_made_of_tok_doc,
                                   doc, idf_dict,
                                   1.5, 0.75, avg_doc_len, doc_len)
            if bm_25 is None:
                return None
            list_of_dict_with_bm25.append(bm_25)

    rank_result = rank_documents(list_of_tf_idf_dict,
                                 'A story about a wizard boy in a tower!', stopwords)
    rank_result = rank_documents(list_of_dict_with_bm25,
                                 'A story about a wizard boy in a tower!', stopwords)

    result = rank_result
    assert result, "Result is None"


if __name__ == "__main__":
    main()
