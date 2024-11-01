"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from math import log


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
    punctuation = """!'"#$%&()*+,-./:;<=>?@[]^_`{|}~1234567890"""r"\""
    for p in punctuation:
        if p in text:
            text = text.replace(p, ' ')
    tokens = text.lower().split()

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
    if (not isinstance(tokens, list)) or (not isinstance(stopwords, list)):
        return None
    if (len(tokens) == 0) or (len(stopwords) == 0):
        return None
    for elem in stopwords:
        if not isinstance(elem, str):
            return None
    for elem in tokens:
        if not isinstance(elem, str):
            return None

    tokens_cleared = []
    for elem in tokens:
        if elem not in stopwords:
            tokens_cleared.append(elem)

    return tokens_cleared


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list):
        return None
    if len(documents) == 0:
        return None
    for document in documents:
        if not isinstance(document, list):
            return None
        if len(document) == 0:
            return None
        for word in document:
            if not isinstance(word, str):
                return None

    uniq_words = set()
    for document in documents:
        for word in document:
            uniq_words.update({word})
    return list(uniq_words)


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
    if not isinstance(vocab, list) or not isinstance(document_tokens, list):
        return None
    if len(vocab) == 0 or len(document_tokens) == 0:
        return None
    for word in vocab:
        if not isinstance(word, str):
            return None
    for word in document_tokens:
        if not isinstance(word, str):
            return None

    tf_dict = {}
    for word in document_tokens:
        tf_dict[word] = document_tokens.count(word) / len(document_tokens)
    for word in vocab:
        if word not in document_tokens:
            tf_dict[word] = 0.0
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
    if not isinstance(vocab, list) or not isinstance(documents, list):
        return None
    if len(vocab) == 0 or len(documents) == 0:
        return None
    for word in vocab:
        if not isinstance(word, str):
            return None
    for document in documents:
        if not isinstance(document, list) or len(document) == 0:
            return None
        for word in document:
            if not isinstance(word, str):
                return None

    idf_dict = {}
    len_documents = len(documents)
    for word in vocab:
        n = 0
        n = sum(n + 1 for lst in documents if word in lst)
        idf_dict[word] = log((len_documents + 0.5 - n) / (n + 0.5))
    return idf_dict


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
    if not isinstance(tf, dict) or not isinstance(idf, dict):
        return None
    if len(tf) == 0 or len(idf) == 0:
        return None

    for key, value in tf.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
    for key, value in idf.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None

    tf_idf_dict = {}
    for key, value in tf.items():
        if key not in idf:
            tf[key] = 0.0
        tf_idf_dict[key] = tf[key] * idf[key]
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
    if not isinstance(vocab, list) or not isinstance(document, list) \
            or not isinstance(idf_document, dict) or \
            not isinstance(k1, float) or not isinstance(b, float):
        return None
    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) \
            or isinstance(doc_len, bool) or \
            len(vocab) == 0 or len(document) == 0 or len(idf_document) == 0:
        return None
    for word in vocab:
        if not isinstance(word, str):
            return None
    for word in document:
        if not isinstance(word, str):
            return None
    for key, value in idf_document.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None

    bm25 = {}
    for word in document:
        if word not in vocab:
            bm25[word] = 0.0
    for key in vocab:
        bm25[key] = ((idf_document[key] * document.count(key) * (k1 + 1))
                     / (document.count(key) + k1 * (1 - b + b * (doc_len / avg_doc_len))))
    return bm25


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
    if not isinstance(indexes, list) or not isinstance(query, str) \
            or not isinstance(stopwords, list):
        return None
    if len(indexes) == 0 or len(query) == 0 or len(stopwords) == 0:
        return None
    for index_dict in indexes:
        if not isinstance(index_dict, dict):
            return None
        for key, values in index_dict.items():
            if not isinstance(key, str) or not isinstance(values, float):
                return None
    for words in stopwords:
        if not isinstance(words, str):
            return None

    query_token = tokenize(query)
    if not isinstance(query_token, list) or len(query_token) == 0:
        return None
    clear_query_token = remove_stopwords(query_token, stopwords)
    if not isinstance(clear_query_token, list) or len(clear_query_token) == 0:
        return None

    doc_rang = []
    rang = 0
    for document in indexes:
        query_token_metric = 0.0
        for token in clear_query_token:
            if token in document:
                query_token_metric += document[token]
        doc_rang.append((rang, query_token_metric))
        rang += 1
    return sorted(doc_rang, key=lambda tup: tup[1], reverse=True)


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


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """


def load_index(file_path: str) -> list[dict[str, float]] | None:
    """
    Load the index from a file.

    Args:
        file_path (str): The path to the file from which to load the index.

    Returns:
        list[dict[str, float]] | None: The loaded index.

    In case of corrupt input arguments, None is returned.
    """


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
