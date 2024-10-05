"""
Laboratory Work #2 starter
"""
import lab_2_retrieval_w_bm25.main as func
import math

# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches,
# too-many-statements, duplicate-code

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
    # assert result, "Result is None"
    docs = []
    vocab = []
    new_vocab = []
    for text in documents:
        tokenize_doc = func.remove_stopwords(func.tokenize(text), stopwords)
        docs.append(tokenize_doc)

    vocab = func.build_vocabulary(docs)
    idf = func.calculate_idf(vocab, docs)

    for text in documents:
        tokenize_doc = func.remove_stopwords(func.tokenize(text), stopwords)
        vocab = func.build_vocabulary(docs)
        tf = func.calculate_tf(vocab, tokenize_doc)
        new_vocab = func.calculate_tf_idf(tf, idf)
        print(new_vocab)




if __name__ == "__main__":
    main()
