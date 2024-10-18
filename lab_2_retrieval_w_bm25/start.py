"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_idf, calculate_tf, calculate_tf_idf,
                                         rank_documents, remove_stopwords, tokenize)

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

    result = None
    tokenized_documents = []
    for document in documents:
        tokens: list[str] = tokenize(document)
        tokenized_documents.append(tokens)

    tokenized_documents_without_stopwords = []
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
        for tokenized_doc in tokenized_documents:
            without_stopwords = remove_stopwords(tokenized_doc, stopwords)
            if not without_stopwords:
                return None
            tokenized_documents_without_stopwords.append(without_stopwords)

    vocab = build_vocabulary(tokenized_documents)
    if not vocab:
        return None

    list_for_tf_idf = []
    for tokenized_doc_without in tokenized_documents_without_stopwords:
        tf = calculate_tf(vocab, tokenized_doc_without)
        idf = calculate_idf(vocab, tokenized_documents_without_stopwords)
        if ((not calculate_tf(vocab, tokenized_doc_without) or
                not calculate_idf(vocab, tokenized_documents_without_stopwords)) or
                not isinstance(tf, dict) or not isinstance(idf, dict)):
            return None
        tf_idf = calculate_tf_idf(tf, idf)
        if not tf_idf:
            return None
        list_for_tf_idf.append(tf_idf)

    avg_len = []
    for doc in tokenized_documents_without_stopwords:
        avg_len.append(len(doc))
    avg_len_doc = sum(avg_len) / len(tokenized_documents_without_stopwords)

    list_for_bm25 = []
    for doc in tokenized_documents_without_stopwords:
        doc_len = len(doc)
        if idf is None:
            return None
        bm25 = calculate_bm25(vocab, doc, idf, 1.5, 0.75, avg_len_doc, doc_len)
        list_for_bm25.append(bm25)

    rank = rank_documents(list_for_tf_idf, "A story about a wizard boy in a tower!", stopwords)
    rank = rank_documents(list_for_bm25, "A story about a wizard boy in a tower!", stopwords)

    result = rank

    assert result, "Result is None"



if __name__ == "__main__":
    main()
