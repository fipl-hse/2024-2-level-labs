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
        document_tokenized = tokenize(doc)
        document_prep = remove_stopwords(document_tokenized, stopwords)
        documents_prep.append(document_prep)

    vocab = build_vocabulary(documents_prep)

    tf_documents = []
    for doc in documents_prep:
        tf = calculate_tf(vocab, doc)
        tf_documents.append(tf)

    idf_documents = calculate_idf(vocab, documents_prep)
    tf_idf_documents = []
    for tf in tf_documents:
        tf_idf = calculate_tf_idf(tf, idf_documents)
        tf_idf_documents.append(tf_idf)

    bm25_documents = []
    avg_doc_len = sum(len(doc) for doc in documents_prep) / len(documents_prep)
    for doc in documents_prep:
        bm25 = calculate_bm25(vocab, doc, idf_documents, 1.5, 0.75, avg_doc_len, len(doc))
        bm25_documents.append(bm25)

    tf_idf_ranked = rank_documents(tf_idf_documents, 'Which fairy tale has Fairy Queen?', stopwords)
    bm25_ranked = rank_documents(bm25_documents, 'Which fairy tale has Fairy Queen?', stopwords)

    print(tf_idf_ranked)
    print(bm25_ranked)
    result = bm25_ranked
    assert result, "Result is None"


if __name__ == "__main__":
    main()
