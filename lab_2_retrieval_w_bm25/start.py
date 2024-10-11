"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable,
# too-many-branches, too-many-statements, duplicate-code

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
        if tokenize(document) is not None:
            new_doc = remove_stopwords(tokenize(document), stopwords)
            if new_doc is not None:
                tokenized_documents.append(new_doc)
                print(new_doc)

    print()

    vocab = build_vocabulary(tokenized_documents)
    idf = calculate_idf(vocab, tokenized_documents)
    if vocab is not None and idf is not None:
        for document in documents:
            new_doc = remove_stopwords(tokenize(document), stopwords)
            if new_doc is not None:
                tf = calculate_tf(vocab, new_doc)
                if tf is not None:
                    print(calculate_tf_idf(tf, idf))

    result = 1
    assert result, "Result is None"


if __name__ == "__main__":
    main()
