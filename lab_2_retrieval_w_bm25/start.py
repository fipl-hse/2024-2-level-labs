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

    docs_tokens = []
    cleared_docs = []
    for document in documents:
        tokens = tokenize(document)
        docs_tokens.append(tokens)
        cleared_docs.append(remove_stopwords(tokens, stopwords))

    vocab = build_vocabulary(docs_tokens)
    if vocab:
        idf = calculate_idf(vocab, docs_tokens)
    else:
        idf = {}

    tf_idf_list = []

    if idf:
        for tokenized_doc in docs_tokens:
            if tokenized_doc:
                tf = calculate_tf(vocab, tokenized_doc)
                if tf:
                    tf_idf = calculate_tf_idf(tf, idf)
                    if tf_idf:
                        tf_idf_list.append(tf_idf)
                        print(tf_idf)



if __name__ == "__main__":
    main()
