"""
Lab 2.

Text retrieval with BM25
"""
import math
import re
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
    return re.findall(r'\b[^\d\W]+\b', text.lower())


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
    if not isinstance(tokens, list) or not isinstance(stopwords, list):
        return None
    if not tokens or any(not isinstance(token, str) for token in tokens):
        return None
    if not stopwords or any(not isinstance(stopword, str) for stopword in stopwords):
        return None

    return [token for token in tokens if token not in stopwords]


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) or not all(isinstance(doc, list) for doc in documents):
        return None

    for document in documents:

        if not isinstance(document, list):
            return None

        if any(not isinstance(token, str) for token in document):
            return None

    unique_tokens = set()

    for document in documents:
        unique_tokens.update(document)

    return list(unique_tokens) or None


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
    if not vocab or not document_tokens:
        return None
    if not isinstance(vocab, list) or any(not isinstance(token, str) for token in vocab):
        return None
    if not isinstance(document_tokens, list) or any(not isinstance(
            token, str) for token in document_tokens):
        return None
    vocab_unique = set(vocab).union(set(document_tokens))
    tf_dict = {token: 0 for token in vocab_unique}
    total_tokens = len(document_tokens)

    for token in document_tokens:
        if token in tf_dict:
            tf_dict[token] += 1

    if total_tokens == 0:
        return {token: 0.0 for token in vocab}

    return {token: count / total_tokens for token, count in tf_dict.items()} if len(
        document_tokens) > 0 else 0.0


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
    if not all(isinstance(word, str) and word.isalpha() for word in vocab):
        return None
    if not all(isinstance(doc, list) for doc in documents):
        return None
    for doc in documents:
        if not all(isinstance(token, str) and token.isalpha() for token in doc):
            return None
    idf_values = {}
    for token in vocab:
        contains_word = 0
        for docs in documents:
            if token in docs:
                contains_word += 1
        idf = math.log((len(documents) - contains_word + 0.5) / (contains_word + 0.5))
        idf_values[token] = idf

        return idf_values



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
    if not isinstance(tf, dict) or any(
            not isinstance(term, str) or not isinstance(freq, float) for term, freq in tf.items()):
        return None
    if not isinstance(idf, dict) or any(
            not isinstance(term, str) or not isinstance(freq, float) for term, freq in idf.items()):
        return None

    tf_idf_values = {}

    for term in tf.keys():
        if term in idf:
            tf_idf_values[term] = tf[term] * idf[term]

    return tf_idf_values or None


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
    if not vocab or not document or not idf_document or not avg_doc_len or not doc_len:
        return None
    if not isinstance(vocab, list) or not all(isinstance(term, str) for term in vocab) or not isinstance(
            document, list) or not all(isinstance(token, str) for token in document) or len(
        vocab) == 0 or len(document) == 0:
        return None

    if not isinstance(idf_document, dict) or len(idf_document) == 0 or any(
            not isinstance(term, str) or not isinstance(score, float) for term, score in idf_document.items()):
        return None
    if not isinstance(k1, float) or not isinstance(b, float) or not isinstance(
            avg_doc_len, float) or not isinstance(doc_len, int) or isinstance(doc_len, bool):
        return None

    tokens = list(set(vocab) | set(document))
    bm25_scores = {}
    for token in tokens:
        token_number = 0

        if token in document:
            token_number += document.count(token)
        if token in idf_document.keys():
            score = idf_document[token] * token_number * (k1 + 1) / (
                    token_number + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            bm25_scores[token] = score
        else:
            bm25_scores[token] = 0.0

    return bm25_scores



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
    if not indexes or not isinstance(indexes, list) or not all(isinstance(index, dict) for index in indexes):
        return None
    if not isinstance(query, str) or not isinstance(
            stopwords, list) or not all(isinstance(token, str) for token in stopwords):
        return None
    if tokenize(query) is None:
        return None
    query_tokens = remove_stopwords(tokenize(query), stopwords)
    if query_tokens is None:
        return None

    doc_scores = []
    for doc_idx, doc in enumerate(indexes):
        score = sum(doc.get(token, 0) for token in query_tokens)
        doc_scores.append((doc_idx, score))
    return sorted(doc_scores, key=lambda x: x[1], reverse=True)


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
