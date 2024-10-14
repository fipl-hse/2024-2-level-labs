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
    doc_tokens = []
    for text in documents:
        tokens = tokenize(text)
        if not isinstance(tokens, list):
            return None
        new_tokens = remove_stopwords(tokens, stopwords)
        if not isinstance(new_tokens,list):
            return None
        doc_tokens.append(new_tokens)
    tf_idf = []
    vocab = build_vocabulary(doc_tokens)
    if not isinstance(vocab, list):
        return None
    idf = calculate_idf(vocab, doc_tokens)
    if not isinstance(idf, dict):
        return None
    for lst in doc_tokens:
        tf = calculate_tf(vocab, lst)
        if not isinstance(tf, dict):
            return None
        tf_idf.append(calculate_tf_idf(tf, idf))
    length_all = 0.0
    for lst in doc_tokens:
        length_all += len(lst)
    avg_doc_len = length_all/len(doc_tokens)
    indexes = []
    for lst in doc_tokens:
        index = calculate_bm25(vocab, lst, idf,
                               1.5, 0.75,
                               avg_doc_len, len(doc_tokens))
        if not isinstance(index, dict):
            return None
        indexes.append(index)
    result = rank_documents(indexes, 'Which fairy tale has Fairy Queen?', stopwords)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
