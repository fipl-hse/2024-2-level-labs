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
    if not text or not isinstance(text, str):
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
    is_not_correct = (not tokens or not isinstance(tokens, list) or
                      not all(isinstance(i, str) for i in tokens) or
                      not stopwords or not isinstance(stopwords, list) or
                      not all(isinstance(i, str) for i in stopwords))
    if is_not_correct:
        return None

    for token in tokens.copy():
        if token in stopwords:
            tokens.remove(token)
    return tokens or None


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    is_not_correct = (not documents or not isinstance(documents, list) or
                      not all(isinstance(i, list) for i in documents) or
                      not all(isinstance(i, str) for j in documents for i in j))
    if is_not_correct:
        return None

    vocab: list[str] = sum(documents, [])
    return vocab or None


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
    is_not_correct = (not vocab or not isinstance(vocab, list) or
                      not all(isinstance(i, str) for i in vocab) or
                      not document_tokens or not isinstance(document_tokens, list) or
                      not all(isinstance(i, str) for i in document_tokens))
    if is_not_correct:
        return None

    tf_dict: dict[str, float] = {term: document_tokens.count(term) / len(document_tokens)
                                 for term in build_vocabulary([vocab, document_tokens])}
    return tf_dict or None


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
    is_not_correct = (not vocab or not isinstance(vocab, list) or
                      not all(isinstance(i, str) for i in vocab) or
                      not documents or not isinstance(documents, list) or
                      not all(isinstance(i, list) for i in documents) or
                      not all(isinstance(i, str) for j in documents for i in j))
    if is_not_correct:
        return None

    freq_dict: dict[str, float] = {}
    for document in documents:
        for term in document:
            num_documents_with_term = sum(1 for document in documents if term in document)
            freq_dict[term] = math.log((len(documents) - num_documents_with_term + 0.5) /
                                       (num_documents_with_term + 0.5))
    return freq_dict or None


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
    is_not_correct = (not tf or not isinstance(tf, dict) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for key, value in tf.items())) or
                      not idf or not isinstance(idf, dict) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for key, value in idf.items())))
    if is_not_correct:
        return None

    tf_idf_dict: dict[str, float] = {term: tf[term] * idf[term] for term in tf}
    return tf_idf_dict or None


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
    is_not_correct = (not vocab or not isinstance(vocab, list) or
                      not all(isinstance(i, str) for i in vocab) or
                      not document or not isinstance(document, list) or
                      not all(isinstance(i, str) for i in document) or
                      not idf_document or not isinstance(idf_document, dict) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for key, value in idf_document.items())) or
                      all(isinstance(i, dict) for i in idf_document.values()) or
                      not k1 or not isinstance(k1, float) or not 1.2 <= k1 <= 2.0 or
                      not b or not isinstance(b, float) or not 0 <= b <= 1 or
                      not avg_doc_len or not isinstance(avg_doc_len, float) or
                      avg_doc_len is None or not doc_len or not isinstance(doc_len, int) or
                      isinstance(doc_len, bool) or doc_len is None)
    if is_not_correct:
        return None

    bm25_dict: dict[str, float] = {}
    for term in build_vocabulary([vocab, document]):
        if term not in vocab:
            bm25_dict[term] = 0.0
            continue
        num_term_occur = document.count(term)
        bm25_dict[term] = (idf_document[term] * ((num_term_occur * (k1 + 1)) /
                                                 (num_term_occur + k1 *
                                                  (1 - b + b * doc_len / avg_doc_len))))
    return bm25_dict or None


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
    is_not_correct = (not indexes or not isinstance(indexes, list) or
                      not all(isinstance(i, dict) for i in indexes) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for i in indexes for key, value in i.items())) or
                      not query or not isinstance(query, str) or
                      not stopwords or not isinstance(stopwords, list) or
                      not all(isinstance(i, str) for i in stopwords))
    if is_not_correct:
        return None

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return None
    query_preprocess = remove_stopwords(tokenized_query, stopwords)
    if not query_preprocess:
        return None

    ranked_document: list[tuple[int, float]] = [(indexes.index(document),
                                                 sum(document[word] for word in document
                                                     if word in query_preprocess))
                                                for document in indexes]
    return sorted(ranked_document, key=lambda x: x[1], reverse=True) or None


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
    is_not_correct = (not vocab or not isinstance(vocab, list) or
                      not all(isinstance(i, str) for i in vocab) or
                      not document or not isinstance(document, list) or
                      not all(isinstance(i, str) for i in document) or
                      not idf_document or not isinstance(idf_document, dict) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for key, value in idf_document.items())) or
                      all(isinstance(i, dict) for i in idf_document.values()) or
                      not alpha or not isinstance(alpha, float) or
                      not k1 or not isinstance(k1, float) or not 1.2 <= k1 <= 2.0 or
                      not b or not isinstance(b, float) or not 0 <= b <= 1 or
                      not avg_doc_len or not isinstance(avg_doc_len, float) or
                      not doc_len or avg_doc_len is None or not isinstance(doc_len, int) or
                      isinstance(doc_len, bool) or doc_len < 0 or doc_len is None)
    if is_not_correct:
        return None

    modified_bm25_dict: dict[str, float] = {}
    for word in vocab:
        if word in idf_document:
            idf = idf_document[word]
            if idf < alpha:
                continue
            num_word_occur = document.count(word)
            modified_bm25_dict[word] = idf * ((num_word_occur * (k1 + 1)) /
                                              (num_word_occur + k1 *
                                               (1 - b + b * doc_len / avg_doc_len)))
    return modified_bm25_dict or None


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    is_not_correct = (not index or not isinstance(index, list) or
                      not all(isinstance(i, dict) for i in index) or
                      not all((isinstance(key, str) and isinstance(value, float)
                               for i in index for key, value in i.items())) or
                      not file_path or not isinstance(file_path, str))
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
        loaded_index: list[dict[str, float]] = json.load(file)
    return loaded_index or None


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
    is_not_correct = (not rank or not isinstance(rank, list) or
                      not all(isinstance(i, int) for i in rank) or
                      not golden_rank or not isinstance(golden_rank, list) or
                      not all(isinstance(i, int) for i in golden_rank) or
                      not len(rank) == len(golden_rank))
    if is_not_correct:
        return None

    n = len(rank)
    spearman_coef: float = 1 - (6 * sum((index - golden_rank.index(number)) ** 2
                                        for index, number in enumerate(rank)
                                        if number in golden_rank)) / (n * (n ** 2 - 1))
    return spearman_coef or None
