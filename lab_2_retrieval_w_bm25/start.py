"""
Laboratory Work #2 starter
"""
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf,
                                         calculate_tf, calculate_tf_idf, rank_documents,
                                         remove_stopwords, tokenize)

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
    result = tokenize(documents[0])
    tokenized_doc = [tokenize(doc) for doc in documents]
    tokenized_doc = [remove_stopwords(doc, stopwords) for doc in tokenized_doc]
    vocabulary = build_vocabulary(tokenized_doc)
    tf_doc = [calculate_tf(vocabulary, doc) for doc in tokenized_doc]
    idf_score = calculate_idf(vocabulary, tokenized_doc)
    tf_idf_doc = [calculate_tf_idf(tf, idf_score) for tf in tf_doc]
    for document in tokenized_doc:

        tf_values = calculate_tf(vocabulary, document)

        tf_idf_values = calculate_tf_idf(tf_values, idf_score)
        tf_idf_doc.append(tf_idf_values)

    for tf_idf_dict in tf_idf_doc:
        print(f'{tf_idf_dict}\n\n')

    avg_doc_len = sum(len(doc) for doc in tokenized_doc) / len(tokenized_doc)
    bm25_doc = [
        calculate_bm25(vocabulary, doc, idf_score, avg_doc_len, len(doc))
        for doc in tokenized_doc
    ]
    query = "A story about a wizard boy in a tower!"
    ranked_bm25 = rank_documents(bm25_doc, query, stopwords)
    print(ranked_bm25)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
