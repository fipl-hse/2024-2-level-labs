"""
Lab 2.

Text retrieval with BM25
"""

import json
import math

# pylint:disable=too-many-arguments, unused-argument


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
    text_new = ''
    for symbol in text:
        if symbol.isalpha() or symbol == ' ':
            right_symbol = symbol.lower()
            text_new += right_symbol
    tokens = text_new.split()
    return tokens


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
    if not isinstance(tokens, list) or not isinstance(stopwords, list) \
            or not all(isinstance(token, str) for token in tokens) \
            or not all(isinstance(word, str) for word in stopwords):
        return None
    tokenize_doc = []
    for word in tokens:
        if word not in stopwords:
            tokenize_doc.append(word)
    return tokenize_doc


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) \
            or not all(isinstance(doc, list) for doc in documents) \
            or not all((isinstance(word, str) for word in doc) for doc in documents):
        return None
    vocab = []
    for tokenize_doc in documents:
        for word in tokenize_doc:
            if word not in vocab:
                vocab.append(word)
    return vocab


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
    if not isinstance(vocab, list) or not isinstance(document_tokens, list) \
            or not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(token, str) for token in document_tokens):
        return None
    tf_dict = {}
    for word in vocab:
        tf_dict[word] = document_tokens.count(word) / abs(len(document_tokens))
    return tf_dict


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
    if not isinstance(vocab, list) or not isinstance(documents, list) \
            or not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(doc, list) for doc in documents) \
            or not all((isinstance(word, str) for word in doc) for doc in documents):
        return None
    idf_dic = {}
    n = 0
    for text in documents:
        n += 1
    for word in vocab:
        amount = 0
        for tokenize_doc in documents:
            if word in tokenize_doc:
                amount += 1
        idf_dic[word] = math.log((n - amount + 0.5) / (amount + 0.5))
    return idf_dic


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
    if not isinstance(tf, dict) or not isinstance(idf, dict) \
            or not all(
        (isinstance(key, str) and isinstance(value, float) for key, value in tf.items())) \
            or not all(
        (isinstance(key, str) and isinstance(value, float) for key, value in idf.items())):
        return None

    tf_idf_dict = {}
    for word in tf:
        tf_idf_dict[word] = tf[word] * idf[word]
    return tf_idf_dict


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
    if not isinstance(idf_document, dict) or not isinstance(k1, float) \
            or not isinstance(b, float) or not isinstance(doc_len, int) \
            or not isinstance(avg_doc_len, float) or not isinstance(vocab, list) \
            or not isinstance(document, list):
        return None
    if not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(token, str) for token in document) or vocab:
        return None
    if not all((isinstance(key, str) and isinstance(value, float) for key, value in
                idf_document.items())):
        return None
    bm25_dict = {}
    for word in vocab:
        n = document.count(word)
        bm25_dict[word] = idf_document[word] * (n * (k1 + 1)) / (
                n + k1 * (1 - b + (b * abs(doc_len) / avg_doc_len)))
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
    if not isinstance(query, str) or not isinstance(stopwords, list) or not isinstance(indexes,
                                                                                       list):
        return None

    if not all(isinstance(stopword, str) for stopword in stopwords) or \
            not all(isinstance(vocab, dict) for vocab in indexes) or \
            not all((isinstance(key, str) and isinstance(value, float) for key, value in vocab) for
                    vocab in indexes):
        return None
    tokenize_str = remove_stopwords(tokenize(query), stopwords)
    result = []
    n = 0
    for text in indexes:
        sum_result = 0
        for word in tokenize_str:
            if word in text.keys():
                sum_result += text[word]
        tuple_result = (n, sum_result)
        n += 1
        result.append(tuple_result)
    result.sort(key=lambda a: a[1], reverse=True)
    return result


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
    if not isinstance(idf_document, dict) or not isinstance(k1, float) \
            or not isinstance(b, float) or not isinstance(doc_len, int) \
            or not isinstance(avg_doc_len, float) or not isinstance(vocab, list) \
            or not isinstance(document, list) or not isinstance(alpha, float):
        return None
    if not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(token, str) for token in document):
        return None
    if not all((isinstance(key, str) and isinstance(value, float) for key, value in
                idf_document.items())):
        return None
    bm25_dict = {}
    for word in vocab:
        n = document.count(word)
        if idf_document[word] > alpha:
            bm25_dict[word] = idf_document[word] * (n * (k1 + 1)) / (
                    n + k1 * (1 - b + (b * abs(doc_len) / avg_doc_len)))
    return bm25_dict


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not isinstance(file_path, str):
        return None
    with open(file_path, "w") as file:
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
    if not isinstance(file_path, str):
        return None
    with open(file_path, 'r') as file:
        result = json.load(file)
    return result


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

