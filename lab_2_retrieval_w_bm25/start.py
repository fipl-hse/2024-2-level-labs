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
    docs_len = 0
    bm25_list = []
    for text in documents:
        tokens = tokenize(text)
        if tokens is None:
            result = None
        cleared_tokens = remove_stopwords(tokens, stopwords)
        if cleared_tokens is None:
            result = None
        doc_tokens.append(cleared_tokens)
        docs_len += len(tokens)
    av_docs_len = docs_len / len(documents)
    if doc_tokens is None:
        result = None
    vocab = build_vocabulary(list(doc_tokens))
    if vocab is None:
        result = None
    idf = calculate_idf(vocab, list(doc_tokens))
    if idf is None:
        result = None
    tf_idf = []
    for lst in doc_tokens:
        if lst is None:
            result = None
        tf = calculate_tf(vocab, lst)
        if tf is None:
            result = None
        tf_idf_dict = calculate_tf_idf(tf, idf)
        if tf_idf_dict is None:
            result = None
        tf_idf.append(tf_idf_dict)

        bm = calculate_bm25(vocab, lst, idf, 1.5, 0.75,
                            av_docs_len, len(lst))
        if bm is None:
            result = None
        bm25_list.append(bm)

    query = 'Which fairy tale has Fairy Queen?'
    if bm25_list is None:
        result = None
    rank = rank_documents(bm25_list, query, stopwords)
    result = rank
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
