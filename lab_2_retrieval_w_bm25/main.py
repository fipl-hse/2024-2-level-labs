"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument

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

    tokenized_list = []

    for elem in text:
        if not elem.isalpha():
            text = text.replace(elem, ' ')
        else:
            if '.' in elem or ',' in elem or '?' in elem or '!' in elem:
                elem = elem.replace('.', ' ')
            if elem[-1] == '.':
                elem = elem.replace('.', '')

    l_text = text.lower()

    for elem in l_text.split():
        if elem not in ['.', ',', '!', '?', ':', ';']:
            tokenized_list.append(elem)

    return tokenized_list

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
    clear_text = []
    if not tokens or not stopwords:
        return None
    if not isinstance(tokens, list) or not all(isinstance(value, str) for value in tokens):
        return None
    if not isinstance(stopwords, list) or not all(isinstance(value, str) for value in stopwords):
        return None

    for token in tokens:
        if token not in stopwords:
            clear_text.append(token)

    return clear_text

def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if (not documents or not isinstance(documents, list) or not all(isinstance(document, list) for document in documents)):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None

    vocabulary = []

    for document in documents:
        for word in document:
            if document.count(word) >= 1:
                vocabulary.append(word)

    return vocabulary


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
    if (not vocab or not isinstance(vocab, list) or not all(isinstance(token, str) for token in vocab)):
        return None

    if (not document_tokens or not isinstance(document_tokens, list) or not all(isinstance(token, str) for token in document_tokens)):
        return None

    freq_vocab = {}

    same_words = [document_tokens, vocab]
    same_words_vocab = build_vocabulary(same_words)

    for word in same_words_vocab:
        if len(document_tokens) != 0 and word is not None and (document_tokens.count(word) / len(document_tokens)) is not None:
            freq_vocab[word] = document_tokens.count(word) / len(document_tokens)

    return freq_vocab


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
    if (not vocab or not isinstance(vocab, list) or not all(isinstance(token, str) for token in vocab)):
        return None
    if not documents or not isinstance(documents, list) or not all(isinstance(document, list) for document in documents):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None

    word_number = 0
    idf_vocab = {}
    documents_number = len(documents)

    for word in vocab:
        for document in documents:
            if word in document:
                word_number += 1
        if ((documents_number - (word_number + 0.5)) / (word_number + 0.5)) > 0 and math.log((documents_number - (word_number + 0.5)) / (word_number + 0.5)) is not None:
            idf = math.log((documents_number + 1 - (word_number + 0.5)) / (word_number + 0.5))
            idf_vocab[word] = idf
            word_number = 0

    if None in idf_vocab:
        return None

    return idf_vocab


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
    if not tf or not idf or not isinstance(tf, dict) or not isinstance(idf, dict):
        return None
    if (not all(isinstance(token, str) for token in tf.keys()) or not all(isinstance(score, float) for score in tf.values())):
        return None
    if (not all(isinstance(token, str) for token in idf.keys()) or not all(isinstance(score, float) for score in idf.values())):
        return None

    tf_idf_vocab = {}

    for word in tf:
        if tf[word] is None or idf[word] is None or word is None:
            return None
        tf_idf_vocab[word] = tf[word] * idf[word]

    if None in tf_idf_vocab:
        return None

    return tf_idf_vocab


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
