"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf,
                                         calculate_tf, calculate_tf_idf, rank_documents,
                                         remove_stopwords, tokenize)


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
    clear_texts = []
    for text in documents:
        tokenized_text = tokenize(text)
        if not isinstance(tokenized_text, list):
            return None
        without_stopwords = remove_stopwords(tokenized_text, stopwords)
        if not isinstance(without_stopwords, list):
            return None
        clear_texts.append(without_stopwords)
    vocabulary = build_vocabulary(clear_texts)
    if not isinstance(vocabulary, list):
        return None
    tf_idf = []
    bm25 = []
    idf = calculate_idf(vocabulary, clear_texts)
    avgdl = sum(len(doc) for doc in documents) / len(documents)
    for text in clear_texts:
        if not isinstance(text, list):
            return None
        tf = calculate_tf(vocabulary, text)
        tf_idf_text = calculate_tf_idf(tf, idf)
        tf_idf.append(tf_idf_text)
        bm25_text = calculate_bm25(vocabulary, text, idf, 1.5, 0.75, avgdl, len(text))
        bm25.append(bm25_text)
    rank_tf_idf = rank_documents(tf_idf, 'Which fairy tale has Fairy Queen?', stopwords)
    rank_bm25 = rank_documents(bm25, 'Which fairy tale has Fairy Queen?', stopwords)
    result = rank_tf_idf, rank_bm25
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
