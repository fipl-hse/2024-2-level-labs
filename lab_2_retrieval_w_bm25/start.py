"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
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
    result = None

    doc_with_tokens = []
    cleared_doc = []
    for document in documents:
        doc_with_tokens.append(tokenize(document))
        cleared_doc.append(remove_stopwords(tokenize(document), stopwords))

    vocab = build_vocabulary(doc_with_tokens)
    doc_with_tf = []
    for doc_with_tokens in doc_with_tf:
        doc_with_tf.append(calculate_tf(vocab,doc_with_tokens))

    idf = calculate_idf(vocab, doc_with_tokens)

    for doc in doc_with_tf:
        result = calculate_tf_idf(doc, idf)
        print(result)

    print(doc_with_tokens[0])
    print(cleared_doc[0])

    result = cleared_doc
    assert result, "Result is None"


if __name__ == "__main__":
    main()
