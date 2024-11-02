"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25,
                                         calculate_bm25_with_cutoff, calculate_idf,
                                         calculate_spearman, calculate_tf, calculate_tf_idf,
                                         load_index, rank_documents, remove_stopwords, save_index,
                                         tokenize)

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

    tokenized_docs = []
    word_count = 0
    for document in documents:
        tokenized_doc = tokenize(document)
        if tokenized_doc:
            word_count += len(tokenized_doc)
            clean_doc = remove_stopwords(tokenized_doc,stopwords)
            if clean_doc:
                tokenized_docs.append(clean_doc)
    print(tokenized_docs)

    vocabulary = build_vocabulary(tokenized_docs)
    if not vocabulary:
        return
    idf = calculate_idf(vocabulary, tokenized_docs)

    tf_idf_all = []
    bm25_all = []
    cutoff_bm_all = []
    avg_len = word_count / len(documents)
    if idf:
        for doc in tokenized_docs:
            if isinstance(doc, list) and vocabulary:
                tf = calculate_tf(vocabulary, doc)
                doc_len = len(doc)
                if tf:
                    tf_idf = calculate_tf_idf(tf,idf)
                    if tf_idf:
                        tf_idf_all.append(tf_idf)
                    bm25 = calculate_bm25(vocabulary, doc, idf, 1.5, 0.75, avg_len, doc_len)
                    if bm25:
                        bm25_all.append(bm25)
                    cutoff_bm25 = calculate_bm25_with_cutoff(vocabulary,doc,idf,0.2,1.5,0.75,
                                                             avg_len, doc_len)
                    if cutoff_bm25:
                        cutoff_bm_all.append(cutoff_bm25)

    print(tf_idf_all)
    print(bm25_all)
    print(cutoff_bm_all)

    print('\n')
    tf_idf_response = rank_documents(tf_idf_all,'Which fairy tale has Fairy Queen?',stopwords)
    bm25_response = rank_documents(bm25_all, 'Which fairy tale has Fairy Queen?', stopwords)
    gold_response = rank_documents(cutoff_bm_all, 'Which fairy tale has Fairy Queen?', stopwords)

    print(tf_idf_response)
    print(bm25_response)
    print(gold_response)

    tf_idf_rank = [info[0] for info in tf_idf_response]
    bm25_rank = [info[0] for info in bm25_response]
    gold_rank = [info[0] for info in gold_response]

    print(tf_idf_rank)
    print(bm25_rank)
    print(gold_rank)

    save_index(cutoff_bm_all,'assets/metrics.json')
    loaded_info = load_index('assets/metrics.json')
    if loaded_info:
        tf_idf_spearman = calculate_spearman(tf_idf_rank,gold_rank)
        bm25_spearman = calculate_spearman(bm25_rank, gold_rank)
        print(tf_idf_spearman)
        print(bm25_spearman)


    result = (tf_idf_spearman, bm25_spearman)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
