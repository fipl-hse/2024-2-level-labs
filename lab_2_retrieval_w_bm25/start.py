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

    tokenized_documents = []
    total_words = 0
    for document in documents:
        tokens = tokenize(document)
        filtered_tokens = remove_stopwords(tokens, stopwords) if tokens else []
        if filtered_tokens:
            tokenized_documents.append(filtered_tokens)
            total_words += len(filtered_tokens)

    vocabulary = build_vocabulary(tokenized_documents)
    idf = calculate_idf(vocabulary, tokenized_documents) if vocabulary else {}

    tf_idf_list = []
    bm25_list = []
    avg_doc_len = total_words / len(tokenized_documents) if tokenized_documents else 0

    for tokenized_doc in tokenized_documents:
        tf = calculate_tf(vocabulary, tokenized_doc)
        if tf:
            tf_idf = calculate_tf_idf(tf, idf)
            if tf_idf:
                tf_idf_list.append(tf_idf)

        doc_len = len(tokenized_doc)
        bm25 = calculate_bm25(vocabulary, tokenized_doc, idf, 1.5, 0.75, avg_doc_len, doc_len)
        if bm25:
            bm25_list.append(bm25)

    print('\nTF-IDF:\n')
    ranked_tf_idf = rank_documents(tf_idf_list, 'Which fairy tale has Fairy Queen?', stopwords)
    print(ranked_tf_idf)

    print('\nBM25:\n')
    ranked_bm25 = rank_documents(bm25_list, 'Which fairy tale has Fairy Queen?', stopwords)
    print(ranked_bm25)

    result = 'something'

    assert result, "Result is None"


if __name__ == "__main__":
    main()
