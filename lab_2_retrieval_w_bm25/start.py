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

    result = None
    prep_documents = []
    for doc in documents:
        token_doc = tokenize(doc)
        if isinstance(token_doc, list):
            no_sw_doc = remove_stopwords(token_doc, stopwords)
            if isinstance(no_sw_doc, list):
                prep_documents.append(no_sw_doc)

    vocab = []
    if len(prep_documents) == len(documents):
        temp_vocab = build_vocabulary(prep_documents)
        if isinstance(temp_vocab, list):
            vocab = temp_vocab[:]

    if len(vocab) > 0:
        idf = calculate_idf(vocab, prep_documents)
        if isinstance(idf, dict):
            avg_len = 0.0
            for prep_doc in prep_documents:
                avg_len += len(prep_doc)
            avg_len /= len(prep_documents)
            tf_idfs = []
            bm25s = []
            for prep_doc in prep_documents:
                tf = calculate_tf(vocab, prep_doc)
                if isinstance(tf, dict):
                    tf_idf = calculate_tf_idf(tf, idf)
                    if isinstance(tf_idf, dict):
                        tf_idfs.append(tf_idf)
                bm25 = calculate_bm25(
                    vocab, prep_doc, idf, avg_doc_len=avg_len, doc_len=len(prep_doc))
                if isinstance(bm25, dict):
                    bm25s.append(bm25)

            if len(tf_idfs) == len(prep_documents) and len(bm25s) == len(prep_documents):
                result = rank_documents(
                    tf_idfs, 'Which fairy tale has Fairy Queen?', stopwords)
                print(result)
                result = rank_documents(
                    bm25s, 'Which fairy tale has Fairy Queen?', stopwords)
                print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
