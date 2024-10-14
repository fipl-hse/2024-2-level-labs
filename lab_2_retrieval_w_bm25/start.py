"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

import lab_2_retrieval_w_bm25.main as funcs


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

    documents = [funcs.remove_stopwords(funcs.tokenize(doc), stopwords) for doc in documents]
    vocab = funcs.build_vocabulary(documents)
    idf = funcs.calculate_idf(vocab, documents)
    avg_len = 0
    for doc in documents:
        avg_len += len(doc)
    avg_len /= len(documents)
    tf_idfs = []
    bm25s = []
    for doc in documents:
        tf = funcs.calculate_tf(vocab, doc)
        tf_idf = funcs.calculate_tf_idf(tf, idf)
        tf_idfs.append(tf_idf)
        bm25 = funcs.calculate_bm25(vocab, doc, idf, avg_doc_len=avg_len, doc_len=len(doc))
        bm25s.append(bm25)

    result = None
    if isinstance(tf_idfs, list) and isinstance(bm25s, list):
        result = funcs.rank_documents(tf_idfs, 'Which fairy tale has Fairy Queen?', stopwords)
        print(result)
        result = funcs.rank_documents(bm25s, 'Which fairy tale has Fairy Queen?', stopwords)
        print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
