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
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    all_tokenized_docs = []
    for text in documents:
        tokenized_document = tokenize(text)
        if tokenized_document is None:
            result = None
            assert result, "Result is None"
        without_stopwords = remove_stopwords(tokenized_document, stopwords)
        if without_stopwords is None:
            result = None
            assert result, "Result is None"
        all_tokenized_docs.append(without_stopwords)
    vocabulary = build_vocabulary(all_tokenized_docs)
    if vocabulary is None:
        result = None
        assert result, "Result is None"
    tf = []
    idf = calculate_idf(vocabulary, all_tokenized_docs)
    for tokenized_doc in all_tokenized_docs:
        tf_of_doc = calculate_tf(vocabulary, tokenized_doc)
        if tf_of_doc is None:
            result = None
            assert result, "Result is None"
        tf.append(tf_of_doc)
    if idf is None:
        result = None
        assert result, "Result is None"
    tf_idf = []
    for tf_doc in tf:
        tf_idf_of_doc = calculate_tf_idf(tf_doc, idf)
        if tf_idf_of_doc is None:
            result = None
            assert result, "Result is None"
        tf_idf.append(tf_idf_of_doc)
    bm25 = []
    avg_len = 0.0
    for tokenized_doc in all_tokenized_docs:
        avg_len += float(len(tokenized_doc))
    avg_len /= float(len(all_tokenized_docs))
    for tokenized_doc in all_tokenized_docs:
        bm25_of_doc = calculate_bm25(vocabulary, tokenized_doc, idf, 1.5, 0.75,
                                     avg_len, len(tokenized_doc))
        if bm25_of_doc is None:
            result = None
            assert result, "Result is None"
        bm25.append(bm25_of_doc)
    rank_tf_idf = rank_documents(tf_idf, 'Which fairy tale has Fairy Queen?', stopwords)
    rank_bm25 = rank_documents(bm25, 'Which fairy tale has Fairy Queen?', stopwords)
    result = rank_tf_idf
    result_2 = rank_bm25
    print(result, '\n', result_2)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
