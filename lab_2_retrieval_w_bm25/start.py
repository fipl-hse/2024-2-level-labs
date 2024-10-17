"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable,
# too-many-branches, too-many-statements, duplicate-code

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

    print('Tokenized_documents \n')

    tokenized_documents = []
    total_number_words = 0
    for document in documents:
        tokenized_document = tokenize(document)
        if tokenized_document is not None:
            total_number_words += len(tokenized_document)
            new_doc = remove_stopwords(tokenized_document, stopwords)
            if new_doc is not None:
                tokenized_documents.append(new_doc)
                print(new_doc)

    print('\n Tf-idf dictionaries \n')

    vocab = build_vocabulary(tokenized_documents)
    if vocab is not None:
        idf = calculate_idf(vocab, tokenized_documents)
    else:
        idf = {}

    if idf is not None:
        tf_idf_list = []
        bm25_list = []
        avg_doc_len = total_number_words / len(documents)
        for document in tokenized_documents:
            if isinstance(document, list):
                tf = calculate_tf(vocab, document)
                doc_len = len(document)
                if tf is not None:
                    tf_idf = calculate_tf_idf(tf, idf)
                    if tf_idf is not None:
                        tf_idf_list.append(tf_idf)
                        print(tf_idf)

                bm25 = calculate_bm25(vocab, document, idf, 1.5, 0.75, avg_doc_len, doc_len)
                if bm25 is not None:
                    bm25_list.append(bm25)

    print('\n Ranking documents using tf-idf:\n')
    print(rank_documents(tf_idf_list, 'Which fairy tale has Fairy Queen?', stopwords))

    print('\n Ranking documents using bm25:\n')
    print(rank_documents(bm25_list, 'Which fairy tale has Fairy Queen?', stopwords))

    result = 'aaaaa why none'
    assert result, "Result is None"


if __name__ == "__main__":
    main()
