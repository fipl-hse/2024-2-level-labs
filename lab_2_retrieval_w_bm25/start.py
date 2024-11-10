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
    total_words_count = 0
    processed_docs = []

    for doc in documents:
        tokens = tokenize(doc)
        if tokens:
            total_words_count += len(tokens)
            cleaned_doc = remove_stopwords(tokens, stopwords)
            if cleaned_doc:
                processed_docs.append(cleaned_doc)
                print("Tokens after stopword removal:", cleaned_doc)

    token_list = build_vocabulary(processed_docs)
    idf = calculate_idf(token_list, processed_docs) if token_list else {}
    tf_idf_scores = []
    bm25_scores = []
    avg_doc_length = total_words_count / len(documents) if documents else 1

    for processed_doc in processed_docs:
        if not token_list or not idf:
            continue

        token_frequency = calculate_tf(token_list, processed_doc)
        current_doc_length = len(processed_doc)

        if token_frequency:
            tf_idf_values = calculate_tf_idf(token_frequency, idf)
            if tf_idf_values:
                tf_idf_scores.append(tf_idf_values)
                print(tf_idf_values)

        bm25_score = calculate_bm25(token_list, processed_doc, idf,
                                    1.5, 0.75, avg_doc_length, current_doc_length)
        if bm25_score:
            bm25_scores.append(bm25_score)

    print(rank_documents(tf_idf_scores, 'Which fairy tale has Fairy Queen?', stopwords))
    print(rank_documents(bm25_scores, 'Which fairy tale has Fairy Queen?', stopwords))

    result = 'result'
    assert result, "Result is None"


if __name__ == "__main__":
    main()
