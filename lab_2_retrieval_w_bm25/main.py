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

    words_without_symb = ''

    for i in text.lower():
        if i.isalpha():
            words_without_symb += i
        if not i.isalpha():
            words_without_symb += ' '

    tokenized_list = words_without_symb.split()

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
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        return None
    if not stopwords or not isinstance(stopwords, list) or \
            not all(isinstance(stopword, str) for stopword in stopwords):
        return None

    words_without_sw = [token for token in tokens if token not in stopwords]

    return words_without_sw or None

def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) or documents is None \
            or not all(isinstance(token, list) for token in documents):
        return None

    unique_words = []

    for doc in documents:
        if not all(isinstance(token, str) for token in doc):
            return None
        for word in doc:
            if word not in unique_words:
                unique_words.append(word)

    return unique_words or None

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
    if not vocab or not document_tokens \
            or not isinstance(vocab, list) or not isinstance(document_tokens, list):
        return None
    if not all(isinstance(word, str) for word in vocab) \
            or not all(isinstance(word, str) for word in document_tokens):
        return None

    result = {}
    words = list(set(vocab + document_tokens))

    for token in words:
        result[token] = document_tokens.count(token) / len(document_tokens)

    return result or None


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
    if not vocab or not documents or not isinstance(vocab, list) or not isinstance(documents, list):
        return None
    if not all(isinstance(word, str) for word in vocab):
        return None
    for document in documents:
        if not isinstance(document, list):
            return None
        for word in document:
            if not isinstance(word, str):
                return None

    idf = {}

    document_counts = {word: 0 for word in vocab}
    for document in documents:
        for word in document:
            if word in vocab and word not in document_counts:
                document_counts[word] += 1

    total_documents = len(documents)
    for word, count in document_counts.items():
        if count > 0:
            idf[word] = math.log(total_documents - count / count)

    return idf


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
    if not all(isinstance(word, str) for word in idf):
        return None

    tf_idf = {}

    for word in tf:
        if word in idf:
            tf_idf[word] = tf[word] * idf[word]

    return tf_idf or None

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
    if not vocab or not isinstance(vocab, list) or \
            not all(isinstance(token, str) for token in vocab):
        return None
    if not document or not isinstance(document, list) or \
            not all(isinstance(token, str) for token in document):
        return None
    if not idf_document or not isinstance(idf_document, dict) or \
            not all(isinstance(token, str) and \
                isinstance(freq, float) for token, freq in idf_document.items()):
        return None
    if not isinstance(k1, float) or not 1.2 <= k1 <= 2 or not isinstance(b, float) or not 0 < b < 1:
        return None
    if not isinstance(avg_doc_len, float) or avg_doc_len is None \
            or not isinstance(doc_len, int) or doc_len is None or isinstance(doc_len, bool):
        return None

    bm25 = {}
    unique_words = list(set(vocab + document))

    for token in unique_words:
        if token not in idf_document:
            bm25[token] = 0.0
        else:
            bm25[token] = ((idf_document[token] * document.count(token) * (k1 + 1))
                           / (document.count(token) + k1 * (1 - b + b * doc_len / avg_doc_len)))

    return bm25 or None


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
