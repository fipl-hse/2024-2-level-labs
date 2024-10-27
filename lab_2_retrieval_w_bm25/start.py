"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf,
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

    #получение списка списков уникальных слов
    doc_tokens = []
    for text in documents:
        tokens = tokenize(text)
        if not isinstance(tokens, list):
            return None
        new_tokens = remove_stopwords(tokens, stopwords)
        if not isinstance(new_tokens,list):
            return None
        doc_tokens.append(new_tokens)

    #подсчёт tf_idf
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
        dict_tf_idf = calculate_tf_idf(tf, idf)
        if not isinstance(dict_tf_idf, dict):
            return None
        tf_idf.append(dict_tf_idf)

    #получение списка словарей bm25
    length_all = 0.0
    for lst in doc_tokens:
        length_all += len(lst)
    avg_doc_len = length_all/len(doc_tokens)
    bm25 = []
    for lst in doc_tokens:
        index = calculate_bm25(vocab, lst, idf,
                               1.5, 0.75,
                               avg_doc_len, len(doc_tokens))
        if not isinstance(index, dict):
            return None
        bm25.append(index)

    #получение списка словарей bm25_cutoff
    bm25_cut = []
    for lst in doc_tokens:
        index = calculate_bm25_with_cutoff(vocab, lst, idf, 0.2,
                               1.5, 0.75,
                               avg_doc_len, len(doc_tokens))
        if not isinstance(index, dict):
            return None
        bm25_cut.append(index)

    #пример сохранения и загрузки в файл
    save_index(bm25_cut, 'assets/metrics.json')
    load_index('assets/metrics.json')

    #получение ранжированных списков
    list_tf_idf = rank_documents(tf_idf, 'Which fairy tale has Fairy Queen?', stopwords)
    list_bm25 = rank_documents(bm25, 'Which fairy tale has Fairy Queen?', stopwords)
    list_bm25_cut = rank_documents(bm25_cut, 'Which fairy tale has Fairy Queen?', stopwords)
    if not isinstance(list_tf_idf, list) or not isinstance(list_bm25, list)\
        or not isinstance(list_bm25_cut, list):
        return None
    rank_tf_idf = []
    rank_bm25 = []
    rank_bm25_cut = []
    for box in list_tf_idf:
        rank_tf_idf.append(box[0])
    for box in list_bm25:
        rank_bm25.append(box[0])
    for box in list_tf_idf:
        rank_bm25_cut.append(box[0])

    #подсчёт коэффициентов Спирмена
    golden_rank = [1, 7, 5, 0, 9, 2, 3, 4, 6, 8]
    spear_tf_idf = calculate_spearman(rank_tf_idf, golden_rank)
    spear_bm25 = calculate_spearman(rank_bm25, golden_rank)
    spear_bm25_cut = calculate_spearman(rank_bm25_cut, golden_rank)
    result = f'sp_tf_idf: {spear_tf_idf}, sp_bm25: {spear_bm25}, sp_bm25_cut: {spear_bm25_cut}'
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
