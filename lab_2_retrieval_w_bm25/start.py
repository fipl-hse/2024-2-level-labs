"""
Laboratory Work #2 starter
"""
import lab_2_retrieval_w_bm25.main as func

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
    docs = []
    for text in documents:
        tokenized_text = func.tokenize(text)
        if tokenized_text is not None:
            tokenized_text = func.remove_stopwords(tokenized_text, stopwords)
        if tokenized_text is not None:
            # tokenized_text = func.remove_stopwords(tokenized_text, stopwords)
            docs.append(tokenized_text)

    vocab = func.build_vocabulary(docs)
    if vocab is not None:
        idf = func.calculate_idf(vocab, docs)
    else:
        idf = {}

    for text in documents:
        tokenized_text = func.tokenize(text)
        if tokenized_text is not None:
            tokenized_text = func.remove_stopwords(tokenized_text, stopwords)
        vocab = func.build_vocabulary(docs)
        if vocab is not None and tokenized_text is not None:
            tf = func.calculate_tf(vocab, tokenized_text)
        else:
            tf = {}
        if tf is not None and idf is not None:
            new_vocab = func.calculate_tf_idf(tf, idf)
        else:
            new_vocab = {}
            assert result, "Result is None"
        print(new_vocab)




if __name__ == "__main__":
    main()
