"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements,duplicate-code


from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_idf, calculate_tf,
                                         calculate_tf_idf, remove_stopwords, tokenize)


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

    for document in documents:
        tokenized_document = tokenize(document)
        tokenized_documents.append(tokenized_document)

    clear_documents = []

    for tok_doc in tokenized_documents:
        clear_documents.append(remove_stopwords(tok_doc, stopwords))

    vocab = build_vocabulary(clear_documents)
    idf = calculate_idf(vocab, clear_documents)

    tf_vocab = []
    tf_idf_vocab = []

    for doc in clear_documents:
        tf = calculate_tf(vocab, doc)
        tf_vocab.append(tf)

    for tf_ in tf_vocab:
        tf_idf = calculate_tf_idf(tf_, idf)
        tf_idf_vocab.append(tf_idf)
        print(tf_idf_vocab)

    result = tf_idf_vocab
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
