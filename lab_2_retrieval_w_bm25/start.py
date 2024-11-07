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

    documents_prep = []
    for doc in documents:
        doc_tokenized = tokenize(doc)
        if doc_tokenized is None:
            result = None
            assert result, "Result is None"
        doc_prep = remove_stopwords(doc_tokenized, stopwords)
        if doc_prep is None:
            result = None
            assert result, "Result is None"
        documents_prep.append(doc_prep)
    if not all(isinstance(doc, list) for doc in documents_prep) or \
            not isinstance(documents_prep, list) or \
            not all(isinstance(token, str) for doc in documents_prep for token in doc):
        result = None
        assert result, "Result is None"

    vocab = build_vocabulary(documents_prep)
    if not isinstance(vocab, list) or not all(isinstance(word, str) for word in vocab):
        result = None
        assert result, "Result is None"

    tf_documents: list[dict[str, float]] = []
    for doc in documents_prep:
        if not isinstance(doc, list) or not all(isinstance(item, str) for item in doc):
            result = None
            assert result, "Result is None"
        tf = calculate_tf(vocab, doc)
        if tf is None or not isinstance(tf, dict) or not all(isinstance(key, str) for key in tf) \
                or not all(isinstance(value, float) for value in tf.values()):
            result = None
            assert result, "Result is None"
        tf_documents.append(tf)

    idf_documents = calculate_idf(vocab, documents_prep)
    tf_idf_documents: list[dict[str, float]] = []
    for tf in tf_documents:
        if not isinstance(idf_documents, dict) \
                or not all(isinstance(key, str) for key in idf_documents) \
                or not all(isinstance(value, float) for value in idf_documents.values()):
            result = None
            assert result, "Result is None"
        tf_idf = calculate_tf_idf(tf, idf_documents)
        if tf_idf is None:
            result = None
            assert result, "Result is None"
        tf_idf_documents.append(tf_idf)

    bm25_documents: list[dict[str, float]] = []
    avg_doc_len = sum(len(doc) for doc in documents_prep) / len(documents_prep)
    for doc in documents_prep:
        if not isinstance(doc, list) or not all(isinstance(item, str) for item in doc):
            result = None
            assert result, "Result is None"
        if idf_documents is None:
            result = None
            assert result, "Result is None"
        bm25 = calculate_bm25(vocab, doc, idf_documents,
                              avg_doc_len=avg_doc_len, doc_len=len(doc))
        if bm25 is None:
            result = None
            assert result, "Result is None"
        bm25_documents.append(bm25)

    tf_idf_ranked = rank_documents(tf_idf_documents, 'Which fairy tale has Fairy Queen?', stopwords)
    if tf_idf_ranked is None:
        result = None
        assert result, "Result is None"
    bm25_ranked = rank_documents(bm25_documents, 'Which fairy tale has Fairy Queen?', stopwords)
    if bm25_ranked is None:
        result = None
        assert result, "Result is None"

    print(tf_idf_ranked)
    print(bm25_ranked)
    result = bm25_ranked
    assert result, "Result is None"


if __name__ == "__main__":
    main()
