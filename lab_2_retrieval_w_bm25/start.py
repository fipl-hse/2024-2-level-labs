"""
Laboratory Work #2 starter
"""

from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_idf, calculate_tf,
                                         calculate_tf_idf, remove_stopwords, tokenize)

# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code


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

    docs_tokenized = []
    for document in documents:
        doc_tokenized = tokenize(document)
        if not isinstance(doc_tokenized, list):
            return
        doc_tokenized = remove_stopwords(doc_tokenized, stopwords)
        if not isinstance(doc_tokenized, list):
            return
        docs_tokenized.append(doc_tokenized)
    print(f'Demo 1: Removing Stop-Words\n{docs_tokenized}')

    vocabulary = build_vocabulary(docs_tokenized)
    if not isinstance(vocabulary, list):
        return
    print('Demo 2: Calculating TF-IDF')
    for doc in docs_tokenized:
        tf = calculate_tf(vocabulary, doc)
        idf = calculate_idf(vocabulary, docs_tokenized)
        if not isinstance(tf, dict) or not isinstance(idf, dict):
            return
        print(calculate_tf_idf(tf, idf))

    result = 1
    assert result, "Result is None"


if __name__ == "__main__":
    main()
