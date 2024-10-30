"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_idf, calculate_tf,
                                         calculate_tf_idf, remove_stopwords, tokenize,
                                         calculate_bm25, rank_documents)


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

    clean_docs = []
    for doc in documents:
        doc_tokenized = tokenize(doc)
        if doc_tokenized is None:
            result = None
            assert result, "Result is None"
        clean_doc = remove_stopwords(doc_tokenized, stopwords)
        if clean_doc is None:
            result = None
            assert result, "Result is None"
        clean_docs.append(clean_doc)
    if clean_docs is None:
        result = None
        assert result, "Result is None"
    for doc in clean_docs:
        if not isinstance(doc, list):
            result = None
            assert result, "Result is None"
        for each in doc:
            if not isinstance(each, str):
                result = None
                assert result, "Result is None"

    vocabulary = build_vocabulary(clean_docs)
    if not isinstance(vocabulary, list):
        result = None
        assert result, "Result is None"
    for voc in vocabulary:
        if not isinstance(voc, str):
            result = None
            assert result, "Result is None"

    tf_documents =  []
    for doc in clean_docs:
        if not isinstance(doc, list):
            result = None
            assert result, "Result is None"
        for each in doc:
            if not isinstance(each, str):
                result = None
                assert result, "Result is None"
        tf_doc = calculate_tf(vocabulary, doc)
        if any((tf_doc is None, not isinstance(tf_doc, dict))):
            result = None
            assert result, "Result is None"
        for k in tf_doc:
            if not isinstance(k, str):
                result = None
                assert result, "Result is None"
            for each in tf_doc.values():
                if not isinstance(each, float):
                    result = None
                    assert result, "Result is None"
        tf_documents.append(tf_doc)

    # simplify
    idf_documents = calculate_idf(vocabulary, clean_docs)
    tf_idf_documents= []
    for tf_doc in tf_documents:
        if not isinstance(idf_documents, dict) \
                or not all(isinstance(key, str) for key in idf_documents) \
                or not all(isinstance(value, float) for value in idf_documents.values()):
            result = None
            assert result, "Result is None"
        tf_idf = calculate_tf_idf(tf_doc, idf_documents)
        if tf_idf is None:
            result = None
            assert result, "Result is None"
        tf_idf_documents.append(tf_idf)

    bm25_documents = []
    avg_doc_len = sum(len(document) for document in clean_docs) / len(
            clean_docs)
    for doc_ in clean_docs:
        if not isinstance(doc_, list) or not all(isinstance(item, str) for item in doc_):
            result = None
            assert result, "Result is None"
        if idf_documents is None:
            result = None
            assert result, "Result is None"
        bm25 = calculate_bm25(vocabulary, doc_, idf_documents,
                              avg_doc_len=avg_doc_len, doc_len=len(doc_))
        if bm25 is None:
            result = None
            assert result, "Result is None"
        bm25_documents.append(bm25)

    # simplify
    query = 'Which fairy tale has Fairy Queen?'
    tf_idf_ranked = rank_documents(tf_idf_documents, query, stopwords)
    if tf_idf_ranked is None:
        result = None
        assert result, "Result is None"
    bm25_ranked = rank_documents(bm25_documents, query, stopwords)
    if bm25_ranked is None:
        result = None
        assert result, "Result is None"

    result = bm25_ranked

    assert result, 'Result in None'

if __name__ == "__main__":
    main()
