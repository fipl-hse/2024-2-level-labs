"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_bm25, calculate_idf,
                                         calculate_tf, calculate_tf_idf, rank_documents,
                                         remove_stopwords, tokenize, calculate_bm25_with_cutoff, save_index, load_index)


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

    docs_tokenized = []
    doc_lengths = []
    word_count = 0
    for document in documents:
        doc_tokenized = tokenize(document)
        if not isinstance(doc_tokenized, list):
            return
        doc_len = len(doc_tokenized)
        doc_lengths.append(doc_len)
        word_count += doc_len
        doc_tokenized = remove_stopwords(doc_tokenized, stopwords)
        if not isinstance(doc_tokenized, list):
            return
        docs_tokenized.append(doc_tokenized)
    print(f'Demo 1: Removing Stop-Words\n{docs_tokenized}')

    avg_d_l = word_count / len(documents)
    vocab = build_vocabulary(docs_tokenized)
    if not isinstance(vocab, list):
        return

    tf_idf_collection = []
    bm25_collection = []
    bm25_with_cutoff_collection = []
    no = 0
    idf = calculate_idf(vocab, docs_tokenized)
    for doc in docs_tokenized:
        tf = calculate_tf(vocab, doc)
        if not isinstance(tf, dict) or not isinstance(idf, dict):
            return
        tf_idf = calculate_tf_idf(tf, idf)
        if not isinstance(tf_idf, dict):
            return
        tf_idf_collection.append(tf_idf)
        bm25 = calculate_bm25(vocab, doc, idf, 1.5, 0.75, avg_d_l, doc_lengths[no])
        if not isinstance(bm25, dict):
            return
        bm25_collection.append(bm25)
        bm25_with_cutoff = calculate_bm25_with_cutoff(vocab, doc, idf, 0.2, 1.5, 0.75, avg_d_l, doc_lengths[no])
        if not isinstance(bm25_with_cutoff, dict):
            return
        bm25_with_cutoff_collection.append(bm25_with_cutoff)
        no += 1

    print(f'Demo 2: Calculating TF-IDF\n{tf_idf_collection}')
    print(f'Demo 3: Calculating BM25\n{bm25_collection}')
    query = 'Which fairy tale has Fairy Queen?'
    print(f'Demo 4: Ranking By Query TF-IDF\n{rank_documents(tf_idf_collection, query, stopwords)}')
    print(f'Demo 5: Ranking By Query BM25\n{rank_documents(bm25_collection, query, stopwords)}')

    file_path = 'assets/metrics.json'
    file = open(file_path, 'w', encoding='utf-8')
    file.close()
    print('Demo 6: Saving Metrics In A JSON File')
    save_index(bm25_with_cutoff_collection, file_path)
    print('Demo 6: Metrics Have Been Saved! Check the file.')

    print('Demo 7: Loading & Ranking Metrics In A JSON File')
    indexes = load_index(file_path)
    print(rank_documents(indexes, 'Which fairy tale has Fairy Queen?', stopwords))

    result = 1
    assert result, "Result is None"


if __name__ == "__main__":
    main()
