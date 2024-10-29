"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf, rank_documents, remove_stopwords,
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
    text_sort = []
    bm25_lis = []
    docs_len = 0
    for text in documents:
        text_new = tokenize(text)
        text_sort.append(remove_stopwords(text_new, stopwords))
        docs_len += len(text_new)
    avg_docs_len = docs_len / len(documents)
    voc = build_vocabulary(text_sort)
    for text in documents:
        new_text = tokenize(text)
        idf = calculate_idf(voc, text_sort)
        bm25_lis.append(calculate_bm25(voc, new_text, idf, 1.5, 0.75,
                                       avg_docs_len, len(new_text)))
    query = 'Which fairy tale has Fairy Queen?'
    rank = rank_documents(bm25_lis, query, stopwords)
    result = rank
    print(result)
    print(documents[0])
    assert result, "Result is None"


if __name__ == "__main__":
    main()
