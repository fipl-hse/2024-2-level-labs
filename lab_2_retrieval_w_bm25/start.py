"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from main import (build_vocabulary, calculate_idf, calculate_tf, calculate_tf_idf, remove_stopwords,
                  tokenize)


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
    tfs = []
    for doc in range(len(documents)):
        documents[doc] = remove_stopwords(tokens=tokenize(documents[doc]), stopwords=stopwords)
    vocab = build_vocabulary(documents=documents)
    for doc in documents:
        tfs.append(calculate_tf(vocab=vocab, document_tokens=doc))
    idf = calculate_idf(vocab=vocab, documents=documents)
    tfs_idf = []
    for tf in tfs:
        tfs_idf.append(calculate_tf_idf(tf=tf, idf=idf))
    result = tfs_idf
    if isinstance(result, list):
        print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
