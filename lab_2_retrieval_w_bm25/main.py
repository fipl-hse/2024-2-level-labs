"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument
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
    text = text.lower()
    for token in text:
        if not token.isalpha() and token != ' ':
            text = text.replace(token, ' ')
    return text.split()


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
    if not isinstance(stopwords, list) or not all(isinstance(stopword, str) for stopword in stopwords):
        return None
    cleared_tokens = []
    for token in tokens:
        if token not in stopwords:
            cleared_tokens.append(token)
    return cleared_tokens

def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) or not all(isinstance(document, list) for document in documents):
        return None
    for document in documents:
        if not all(isinstance(word, str) for word in document):
            return None

    vocabular = []
    for document in documents:
        for word in document:
            if word not in vocabular:
                vocabular.append(word)
    return vocabular

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

    if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
        return None
    if not isinstance(document_tokens, list) or not all(isinstance(token, str) for token in document_tokens):
        return None

    result = {}
    for word in set(vocab) | set(document_tokens):
        result[word] = document_tokens.count(word) / len(document_tokens)
    return result

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

    if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
        return None
    if not isinstance(documents, list) or not all(isinstance(doc, list) for doc in documents) or \
            not all(isinstance(item, str) for doc in documents for item in doc):
        return None

    total_documents = len(documents)
    idf = {}
    for word in vocab:
        doc_has_word_count = 0
        for document in documents:
            if word in document:
                doc_has_word_count += 1
        idf[word] = log((total_documents - doc_has_word_count + 0.5) / (doc_has_word_count + 0.5))
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

    if not isinstance(tf, dict) or not all(isinstance(key, str) for key in tf) or \
            not all(isinstance(value, float) for value in tf.values()):
        return None
    if not not isinstance(idf, dict) or not all(isinstance(key, str) for key in idf) or \
            not all(isinstance(value, float) for value in idf.values()):
        return None

    tf_idf = {}
    for word in tf:
        tf_idf[word] = tf[word] * idf[word]
    return tf_idf

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
    if not vocab or not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
        return None
    if not document or not isinstance(document, list) \
            or not all(isinstance(item, str) for item in document):
        return None
    if not idf_document or not isinstance(idf_document, dict) \
            or not all(isinstance(key, str) for key in idf_document) \
            or not all(isinstance(value, float) for value in idf_document.values()):
        return None
    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) \
            or isinstance(doc_len, bool) or not isinstance(k1, float) or not isinstance(b, float):
        return None

    bm25 = {}
    for word in set(vocab) | set(document):
        if word in idf_document:
            word_count = document.count(word)
            bm25[word] = idf_document[word] * ((word_count * (k1 + 1)) / (
                    word_count + k1 * (1 - b + (b * doc_len / avg_doc_len))))
        else: bm25[word] = 0.0
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

    if not indexes or not isinstance(indexes, list) \
            or not all(isinstance(item, dict) for item in indexes) or \
            not all(isinstance(key, str) for item in indexes for key in item) or \
            not all(isinstance(value, float) for item in indexes for value in item.values()):
        return None
    if not isinstance(query, str) or not isinstance(stopwords, list) or \
            not all(isinstance(item, str) for item in stopwords):
        return None

    tokenized_query = tokenize(query)
    if tokenized_query is None:
        return None
    preprocessed_query = remove_stopwords(tokenized_query, stopwords)
    if preprocessed_query is None:
        return None

    result = []
    for i, document in enumerate(indexes):
        result.append((i, sum(document[word] if word in document else 0
                              for word in preprocessed_query)))
    return sorted(result, reverse=True, key=lambda tuple_: tuple_[1])

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
