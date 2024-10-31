"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import tokenize, remove_stopwords, build_vocabulary, calculate_tf, calculate_idf, calculate_tf_idf


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
        tfa = []
        for document in range(len(documents)):
            documents[document] = remove_stopwords(tokens=tokenize(documents[document]), stopwords=stopwords)
        vocab = build_vocabulary(documents=documents)
        for document in documents:
            tfa.append(calculate_tf(vocab=vocab, document_tokens=document))
        idf = calculate_idf(vocab=vocab, documents=documents)
        tfa_idf = []
        for tf in tfa:
            tfa_idf.append(calculate_tf_idf(tf=tf, idf=idf))
        result = tfa_idf
        if isinstance(result, list):
            print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
