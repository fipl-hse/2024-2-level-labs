"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import build_vocabulary, remove_stopwords, tokenize
from lab_2_retrieval_w_bm25.main import (build_vocabulary,calculate_idf,
                                         calculate_spearman, calculate_tf, calculate_tf_idf,
                                         load_index, rank_documents, remove_stopwords, save_index,
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
    result = None

    documents_preprocessed = []
    for document in documents:
        document_tokenized = tokenize(document)
        if document_tokenized is None:
            result = None
            assert result, "Result is None"
        document_preprocessed = remove_stopwords(document_tokenized, stopwords)
        if document_preprocessed is None:
            result = None
            assert result, "Result is None"
        documents_preprocessed.append(document_preprocessed)

    vocabulary = build_vocabulary(documents_preprocessed)

    tf_documents = []
    for doc in documents_preprocessed:
        tf = calculate_tf(vocabulary, doc)
        tf_documents.append(tf)

    idf_documents = calculate_idf(vocabulary, documents_preprocessed)

    tf_idf_doc = []
    for tf in tf_documents:
        tf_idf = calculate_tf_idf(tf, idf_documents)
        tf_idf_doc.append(tf_idf)













    assert result, "Result is None"


if __name__ == "__main__":
    main()
