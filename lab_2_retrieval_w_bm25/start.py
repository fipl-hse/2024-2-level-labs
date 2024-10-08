"""
Laboratory Work #2 starter
"""
from lab_2_retrieval_w_bm25.main import (tokenize, remove_stopwords, build_vocabulary,
                                         calculate_tf, calculate_idf, calculate_tf_idf,
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

    clear_documents = []
    for document in documents:
        tokenized_document = tokenize(document)
        if not tokenized_document:
            return None
        clear_document = remove_stopwords(tokenized_document, stopwords)
        if not clear_document:
            return None
        clear_documents.append(clear_document)

    vocab = build_vocabulary(clear_documents)
    if not vocab:
        return None
    idf = calculate_idf(vocab, clear_documents)
    if not idf:
        return None

    tf_idf_list = []
    bm25_list = []
    for text in clear_documents:
        tf = calculate_tf(vocab, text)
        if not tf:
            return None
        len_text = len(text)
        tf_idf_list.append(calculate_tf_idf(tf, idf))
        bm25_list.append(calculate_bm25(vocab, text, idf, 1.5, 0.75,
                                        sum(len(document) for document
                                            in clear_documents) / len_text, len_text))

    ranked_tf_idf = rank_documents(tf_idf_list, 'Which fairy tale has Fairy Queen?', stopwords)
    ranked_bm25 = rank_documents(bm25_list, 'Which fairy tale has Fairy Queen?', stopwords)

    result = (ranked_tf_idf, ranked_bm25)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
