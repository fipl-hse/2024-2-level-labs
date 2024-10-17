"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument
import json
import math


def tokenize(text: str) -> list[str] | None:
    """
    Tokenize the input text into lowercase words without punctuation, digits and other symbols.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list[str] | None: A list of words from the text.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str):
        return None

    for symbol in text:
        if not symbol.isalpha():
            text = text.replace(symbol, ' ')
    return text.lower().split()


def remove_stopwords(tokens: list[str], stopwords: list[str]) -> list[str] | None:
    """
    Remove stopwords from the list of tokens.

    Args:
        tokens (list[str]): List of tokens.
        stopwords (list[str]): List of stopwords.

    Returns:
        list[str] | None: Tokens after removing stopwords.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not tokens or
                      not isinstance(tokens, list) or
                      not all(isinstance(token, str) for token in tokens) or

                      not stopwords or
                      not isinstance(stopwords, list) or
                      not all(isinstance(word, str) for word in stopwords))
    if is_not_correct:
        return None

    for token in tokens.copy():
        if token in stopwords:
            tokens.remove(token)
    return tokens


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not documents or
                      not isinstance(documents, list) or
                      not all(isinstance(document, list) for document in documents) or
                      not all(isinstance(term, str) for document in documents for term in document))
    if is_not_correct:
        return None

    return list(set(sum(documents, [])))


def calculate_tf(vocab: list[str], document_tokens: list[str]) -> dict[str, float] | None:
    """
    Calculate term frequency for the given tokens based on the vocabulary.

    Args:
        vocab (list[str]): Vocabulary list.
        document_tokens (list[str]): Tokenized document.

    Returns:
        dict[str, float] | None: Mapping from vocabulary terms to their term frequency.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not vocab or
                      not isinstance(vocab, list) or
                      not all(isinstance(term, str) for term in vocab) or

                      not document_tokens or
                      not isinstance(document_tokens, list) or
                      not all(isinstance(token, str) for token in document_tokens))
    if is_not_correct:
        return None

    document_tokens_length = len(document_tokens)
    built_vocabulary = build_vocabulary([vocab, document_tokens])
    return {term: document_tokens.count(term) / document_tokens_length
            for term in built_vocabulary}


def calculate_idf(vocab: list[str], documents: list[list[str]]) -> dict[str, float] | None:
    """
    Calculate inverse document frequency for each term in the vocabulary.

    Args:
        vocab (list[str]): Vocabulary list.
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        dict[str, float] | None: Mapping from vocabulary terms to its IDF scores.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not vocab or
                      not isinstance(vocab, list) or
                      not all(isinstance(term, str) for term in vocab) or

                      not documents or
                      not isinstance(documents, list) or
                      not all(isinstance(document, list) for document in documents) or
                      not all(isinstance(term, str) for document in documents for term in document))
    if is_not_correct:
        return None

    num_documents = len(documents)
    freq_dict = {}
    for document in documents:
        for term in document:
            num_documents_with_term = sum(1 for document in documents if term in document)
            freq_dict[term] = math.log((num_documents - num_documents_with_term + 0.5) /
                                       (num_documents_with_term + 0.5))
    return freq_dict


def calculate_tf_idf(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float] | None:
    """
    Calculate TF-IDF scores for a document.

    Args:
        tf (dict[str, float]): Term frequencies for the document.
        idf (dict[str, float]): Inverse document frequencies.

    Returns:
        dict[str, float] | None: Mapping from terms to their TF-IDF scores.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not tf or
                      not isinstance(tf, dict) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for key, value in tf.items())) or

                      not idf or
                      not isinstance(idf, dict) or
                      not all(isinstance(key, str) and isinstance(value, float)
                              for key, value in idf.items()))
    if is_not_correct:
        return None

    return {term: tf[term] * idf[term] for term in tf}


def calculate_bm25(
    vocab: list[str],
    document: list[str],
    idf_document: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float | None = None,
    doc_len: int | None = None,
) -> dict[str, float] | None:
    """
    Calculate BM25 scores for a document.

    Args:
        vocab (list[str]): Vocabulary list.
        document (list[str]): Tokenized document.
        idf_document (dict[str, float]): Inverse document frequencies.
        k1 (float): BM25 parameter.
        b (float): BM25 parameter.
        avg_doc_len (float | None): Average document length.
        doc_len (int | None): Length of the document.

    Returns:
        dict[str, float] | None: Mapping from terms to their BM25 scores.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not vocab or
                      not isinstance(vocab, list) or
                      not all(isinstance(term, str) for term in vocab) or

                      not document or
                      not isinstance(document, list) or
                      not all(isinstance(term, str) for term in document) or

                      not idf_document or
                      not isinstance(idf_document, dict) or
                      not all(isinstance(key, str) and isinstance(value, float)
                              for key, value in idf_document.items()) or

                      not isinstance(k1, float) or

                      not isinstance(b, float) or

                      not isinstance(avg_doc_len, float) or

                      not isinstance(doc_len, int) or
                      isinstance(doc_len, bool))
    if is_not_correct or avg_doc_len is None or doc_len is None:
        return None

    bm25_dict = {}
    built_vocabulary = build_vocabulary([vocab, document])
    for term in built_vocabulary:
        if term not in idf_document:
            bm25_dict[term] = 0.0
            continue
        num_term_occur = document.count(term)
        bm25_dict[term] = (idf_document[term] * num_term_occur * (k1 + 1) /
                           (num_term_occur + k1 * (1 - b + b * doc_len / avg_doc_len)))
    return bm25_dict


def rank_documents(
    indexes: list[dict[str, float]], query: str, stopwords: list[str]
) -> list[tuple[int, float]] | None:
    """
    Rank documents for the given query.

    Args:
        indexes (list[dict[str, float]]): List of BM25 or TF-IDF indexes for the documents.
        query (str): The query string.
        stopwords (list[str]): List of stopwords.

    Returns:
        list[tuple[int, float]] | None: Tuples of document index and its score in the ranking.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not indexes or
                      not isinstance(indexes, list) or
                      not all(isinstance(index, dict) for index in indexes) or

                      not isinstance(stopwords, list))
    if is_not_correct:
        return None

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return None
    query_preprocess = remove_stopwords(tokenized_query, stopwords)
    if not query_preprocess:
        return None

    ranked_documents = []
    for document in indexes:
        sum_indexes = 0.0
        document_index = indexes.index(document)
        for word in document:
            if word in query_preprocess:
                sum_indexes += document[word]
        ranked_documents.append((document_index, sum_indexes))
    return sorted(ranked_documents, key=lambda x: x[1], reverse=True)


def calculate_bm25_with_cutoff(
    vocab: list[str],
    document: list[str],
    idf_document: dict[str, float],
    alpha: float,
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float | None = None,
    doc_len: int | None = None,
) -> dict[str, float] | None:
    """
    Calculate BM25 scores for a document with IDF cutoff.

    Args:
        vocab (list[str]): Vocabulary list.
        document (list[str]): Tokenized document.
        idf_document (dict[str, float]): Inverse document frequencies.
        alpha (float): IDF cutoff threshold.
        k1 (float): BM25 parameter.
        b (float): BM25 parameter.
        avg_doc_len (float | None): Average document length.
        doc_len (int | None): Length of the document.

    Returns:
        dict[str, float] | None: Mapping from terms to their BM25 scores with cutoff applied.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not vocab or
                      not isinstance(vocab, list) or
                      not all(isinstance(term, str) for term in vocab) or

                      not document or
                      not isinstance(document, list) or
                      not all(isinstance(term, str) for term in document) or

                      not idf_document or
                      not isinstance(idf_document, dict) or
                      not all(isinstance(key, str) and isinstance(value, float)
                              for key, value in idf_document.items()) or

                      not isinstance(alpha, float) or

                      not isinstance(k1, float) or

                      not isinstance(b, float) or

                      not isinstance(avg_doc_len, float) or

                      not isinstance(doc_len, int) or
                      isinstance(doc_len, bool) or
                      doc_len < 0)
    if is_not_correct or avg_doc_len is None or doc_len is None:
        return None

    modified_bm25_dict = {}
    built_vocabulary = build_vocabulary([vocab, document])
    for term in built_vocabulary:
        if term in idf_document:
            idf = idf_document[term]
            if idf < alpha:
                continue
            num_term_occur = document.count(term)
            modified_bm25_dict[term] = (idf * num_term_occur * (k1 + 1) /
                                        (num_term_occur + k1 * (1 - b + b * doc_len / avg_doc_len)))
    return modified_bm25_dict


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    is_not_correct = (not index or
                      not isinstance(index, list) or

                      not file_path or
                      not isinstance(file_path, str))
    if is_not_correct:
        return None

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(index, file, indent=4)
    return None


def load_index(file_path: str) -> list[dict[str, float]] | None:
    """
    Load the index from a file.

    Args:
        file_path (str): The path to the file from which to load the index.

    Returns:
        list[dict[str, float]] | None: The loaded index.

    In case of corrupt input arguments, None is returned.
    """
    if not file_path or not isinstance(file_path, str):
        return None

    with open(file_path, 'r', encoding='utf-8') as file:
        loaded_index = json.load(file)
    if (not loaded_index or
            not isinstance(loaded_index, list) or
            not all(isinstance(index_dict, dict) for index_dict in loaded_index) or
            not all((isinstance(key, str) and isinstance(value, float)
                     for index_dict in loaded_index for key, value in index_dict.items()))):
        return None
    return loaded_index


def calculate_spearman(rank: list[int], golden_rank: list[int]) -> float | None:
    """
    Calculate Spearman's rank correlation coefficient between two rankings.

    Args:
        rank (list[int]): Ranked list of document indices.
        golden_rank (list[int]): Golden ranked list of document indices.

    Returns:
        float | None: Spearman's rank correlation coefficient.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not isinstance(rank, list) or
                      not all(isinstance(i, int) for i in rank) or

                      not golden_rank or
                      not isinstance(golden_rank, list) or
                      not all(isinstance(i, int) for i in golden_rank) or
                      not len(rank) == len(golden_rank))
    if is_not_correct:
        return None

    rank_difference = 0.0
    rank_length = len(rank)
    for index, number in enumerate(rank):
        if number in golden_rank:
            rank_difference += (index - golden_rank.index(number)) ** 2
    return 1 - (6 * rank_difference) / (rank_length * (rank_length ** 2 - 1))
