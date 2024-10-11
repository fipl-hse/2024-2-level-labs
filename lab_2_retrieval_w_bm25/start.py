"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

import main as funcs


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
    tf_idf = funcs.calculate_tf_idf(funcs.calculate_tf(vocab, documents[0]), funcs.calculate_idf(vocab, documents))
    result = tf_idf
    assert result, "Result is None"
    print(result)


if __name__ == "__main__":
    main()
