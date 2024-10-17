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
    voc = []
    tokens = []
    for text in documents:
        tokens.append(tokenize(text))
        voc.append(remove_stopwords(tokenize(text), stopwords))
    a = build_vocabulary(voc)
    # print(a)
    tf = calculate_tf(a, remove_stopwords(tokenize(documents[1]), stopwords))
    idf = calculate_idf(a, tokens)
    result = calculate_tf_idf(tf, idf)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
