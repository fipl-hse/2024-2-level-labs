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
        if not isinstance(tokens, list):
            return None
        cleared_tokens = remove_stopwords(tokens, stopwords)
        doc_tokens.append(cleared_tokens)
        docs_len += len(tokens)
    av_docs_len = docs_len / len(documents)
    if doc_tokens is None or not isinstance(doc_tokens, list):
        return None
    vocab = build_vocabulary(list(doc_tokens))
    if vocab is None or not isinstance(vocab, list):
        return None
    idf = calculate_idf(vocab, list(doc_tokens))
    if not isinstance(idf, dict):
        return
    tf_idf = []
    for lst in doc_tokens:
        if lst is None or not isinstance(lst, list):
            return None
        tf = calculate_tf(vocab, lst)
        if not isinstance(tf, dict):
            return
        tf_idf_dict = calculate_tf_idf(tf, idf)
        if not isinstance(tf_idf_dict, dict):
            return
        tf_idf.append(tf_idf_dict)

        if lst is None or not isinstance(lst, list):
            return None
        bm = calculate_bm25(vocab, lst, idf, 1.5, 0.75,
                            av_docs_len, len(lst))
        if bm is None:
            return
        bm25_list.append(bm)

    query = 'Which fairy tale has Fairy Queen?'
    if bm25_list is None:
        return
    rank = rank_documents(bm25_list, query, stopwords)
    result = rank
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
