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
        tokenized_documents = []
        for document in documents:
            tokens = tokenize(document)
            if tokens:
                tokens = remove_stopwords(tokens, stopwords)
                tokenized_documents.append(tokens)
            else:
                return None
        vocab = build_vocabulary(tokenized_documents)
        idf = calculate_idf(vocab, tokenized_documents)
        tf_idf_l = []
        for document_tokens in tokenized_documents:
            tf = calculate_tf(document_tokens, vocab)
            if not tf:
                return None
            else:
                tf_idf = calculate_tf_idf(tf, idf)
                if not tf_idf:
                    return None
                else:
                    tf_idf_l.append(tf_idf)
        document_number = 1
        for tf_idf in tf_idf_l:
            print(f"TF-IDF для документа {document_number}:")
            print(tf_idf)
            document_number += 1
    result = tf_idf
    assert result, "Result is None"


if __name__ == "__main__":
    main()
