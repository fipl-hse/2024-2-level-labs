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

    tokenized_documents = []

    for document in documents:
        tokenized_documents.append(tokenize(document))

    clear_tokenized_documents = []

    for tokenized_document in tokenized_documents:
        clear_document = remove_stopwords(tokenized_document, stopwords)
        clear_tokenized_documents.append(clear_document)

    for clear_document in clear_tokenized_documents:
        if not isinstance(clear_document, list):
            return None
        return calculate_tf_idf(calculate_tf(build_vocabulary(clear_tokenized_documents), clear_document),
              calculate_idf(build_vocabulary(clear_tokenized_documents), clear_document))

    result = None
    assert result, "Result is None"



if __name__ == "__main__":
    main()
