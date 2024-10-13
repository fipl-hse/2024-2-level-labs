"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument
from json import dump, load
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

    for char in text:
        if not char.isalpha() and char != ' ':
            text = text.replace(char, ' ')
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
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens) or \
            not tokens:
        return None
    if not isinstance(stopwords, list) or not all(isinstance(word, str) for word in stopwords) or \
            not stopwords:
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
    if not isinstance(documents, list) or \
            not all(isinstance(document, list) for document in documents) or \
            not documents:
        return None
    for document in documents:
        if not all(isinstance(item, str) for item in document):
            return None

    result = set()
    for doc in documents:
        result |= set(doc)
    return list(result)


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
    if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab) or \
            not vocab or not isinstance(document_tokens, list) or \
            not all(isinstance(token, str) for token in document_tokens) or not document_tokens:
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
    if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab) or \
            not isinstance(documents, list) or not all(isinstance(doc, list) for doc in documents) \
            or not all(isinstance(item, str) for doc in documents for item in doc) or \
            not vocab or not documents:
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
    if not tf or not idf or not isinstance(tf, dict) or not isinstance(idf, dict) or \
            not all(isinstance(key, str) for key in tf) or \
            not all(isinstance(key, str) for key in idf) or \
            not all(isinstance(value, float) for value in tf.values()) or \
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
    if not vocab or not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab)\
            or not document or not isinstance(document, list) \
            or not all(isinstance(item, str) for item in document) \
            or not idf_document or not isinstance(idf_document, dict) \
            or not all(isinstance(key, str) for key in idf_document) \
            or not all(isinstance(value, float) for value in idf_document.values()) \
            or not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) \
            or isinstance(doc_len, bool) or not isinstance(k1, float) or not isinstance(b, float):
        return None

    bm25 = {}
    for word in set(vocab) | set(document):
        if word in idf_document:
            word_count = document.count(word)
            bm25[word] = idf_document[word] * ((word_count * (k1 + 1)) / (
                    word_count + k1 * (1 - b + (b * doc_len / avg_doc_len))))
        else:
            bm25[word] = 0.0
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
            not all(isinstance(value, float) for item in indexes for value in item.values()) or \
            not isinstance(query, str) or not isinstance(stopwords, list) or \
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
    if not vocab or not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab)\
            or not document or not isinstance(document, list) \
            or not all(isinstance(item, str) for item in document) or not idf_document \
            or not isinstance(idf_document, dict) \
            or not all(isinstance(key, str) for key in idf_document) \
            or not all(isinstance(value, float) for value in idf_document.values()) \
            or not isinstance(alpha, float) or not isinstance(k1, float) \
            or not isinstance(b, float) or not isinstance(avg_doc_len, float) \
            or not isinstance(doc_len, int) or isinstance(doc_len, bool) or doc_len < 0:
        return None

    bm25_with_cutoff = {}
    for word in vocab:
        if word in idf_document and idf_document[word] >= alpha:
            word_count = document.count(word)
            bm25_with_cutoff[word] = idf_document[word] * ((word_count * (k1 + 1)) / (
                    word_count + k1 * (1 - b + (b * doc_len / avg_doc_len))))
    return bm25_with_cutoff


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not index or not isinstance(index, list) \
            or not all(isinstance(item, dict) for item in index) or \
            not all(isinstance(key, str) for item in index for key in item) or \
            not all(isinstance(value, float) for item in index for value in item.values()) or \
            not isinstance(file_path, str) or not file_path:
        return None

    with open(file_path, 'w', encoding='utf-8') as file:
        dump(index, file)
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
        index: list[dict[str, float]] = load(file)
    return index


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
    if not rank or not isinstance(rank, list) or not all(isinstance(item, int) for item in rank) or\
            not golden_rank or not isinstance(golden_rank, list) or \
            not all(isinstance(item, int) for item in golden_rank) or \
            len(rank) != len(golden_rank):
        return None

    n = len(rank)
    rank_differences = 0
    for item in rank:
        if item in rank and item in golden_rank:
            rank_differences += (golden_rank.index(item) - rank.index(item)) ** 2
    return 1 - (6 * rank_differences) / (n * (n**2 - 1))
