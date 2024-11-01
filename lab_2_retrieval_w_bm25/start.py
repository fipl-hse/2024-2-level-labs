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
    doc_tokens = []
    for text in documents:
        tokens = tokenize(text)
        if not isinstance(tokens, list):
            return None
        needful_tokens = remove_stopwords(tokens, stopwords)
        if not isinstance(needful_tokens, list):
            return None
        doc_tokens.append(needful_tokens)

    tf_idf_res = []
    vocab = build_vocabulary(doc_tokens)
    if not isinstance(vocab, list):
        return None
    idf = calculate_idf(vocab, doc_tokens)
    assert isinstance(idf, dict)
    for item in doc_tokens:
        tf = calculate_tf(vocab, item)
        assert isinstance(tf, dict)
        union_dict = calculate_tf_idf(tf, idf)
        if not isinstance(union_dict, dict):
            return None
        tf_idf_res.append(union_dict)


        length = 0.0
        length += len(item)
        avg_len = length / len(doc_tokens)
        bm25 = []
        bm25_res = calculate_bm25(vocab, item, idf, 1.5, 0.75, avg_len, len(doc_tokens))
        if not isinstance(bm25_res, dict):
            return None
        bm25.append(bm25_res)

        tf_idf_work = rank_documents(tf_idf_res, 'Which fairy tale has Fairy Queen?', stopwords)
        bm25_work = rank_documents(bm25, 'Which fairy tale has Fairy Queen?', stopwords)
        if (not isinstance(tf_idf_work, list) or not isinstance(bm25_work, list)
                or not tf_idf_work or not bm25_work):
            return None
        rang_1 = []
        rang_2 = []
        for i in tf_idf_work:
            rang_1.append(i[0])
        for i in bm25_work:
            rang_2.append(i[0])

    result = f'Метрика TF-IDF: {rang_1}, метрика BM25: {rang_2}'
    assert result, "Result is None"


if __name__ == "__main__":
    main()
