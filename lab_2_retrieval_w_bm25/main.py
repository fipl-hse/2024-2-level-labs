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

    text = text.lower()
    tokens = []
    word = []
    for character in text:
        if character.isalpha():
            word.append(character)
        elif word:
            tokens.append(''.join(word))
            word = []
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
    if (not tokens or not isinstance(tokens, list) or
            not all(isinstance(token, str) for token in tokens)):
        return None
    if (not stopwords or not isinstance(stopwords, list) or
            not all(isinstance(word, str) for word in stopwords)):
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
    if (not documents or not isinstance(documents, list) or
            not all(isinstance(document, list) for document in documents)):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None

    vocab = []
    for document in documents:
        for token in document:
            if token not in vocab:
                vocab.append(token)
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
    if (not vocab or not isinstance(vocab, list) or
            not all(isinstance(token, str) for token in vocab)):
        return None
    if (not document_tokens or not isinstance(document_tokens, list) or
            not all(isinstance(token, str) for token in document_tokens)):
        return None

    tf = {}
    common_tokens = [vocab, document_tokens]
    common_vocab = build_vocabulary(common_tokens)
    document_len = len(document_tokens)
    for token in common_vocab:
        tf[token] = document_tokens.count(token)/document_len
    return tf


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
    if (not vocab or not isinstance(vocab, list) or
            not all(isinstance(token, str) for token in vocab)):
        return None
    if not documents or not isinstance(documents, list) or not all(
            isinstance(document, list) for document in documents):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None

    idf = {}
    document_count = len(documents)
    for token in vocab:
        docs_with_tok = sum(1 for document in documents if token in document)
        idf[token] = log((document_count-docs_with_tok+0.5)/(docs_with_tok+0.5))
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
    if (not all(isinstance(token, str) for token in tf.keys()) or
            not all(isinstance(score, float) for score in tf.values())):
        return None
    if (not all(isinstance(token, str) for token in idf.keys()) or
            not all(isinstance(score, float) for score in idf.values())):
        return None

    tf_idf = {}
    for token in tf:
        tf_idf[token] = tf[token]*idf[token]
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
    if (not vocab or not isinstance(vocab, list) or
            not all(isinstance(token, str) for token in vocab)):
        return None
    if (not document or not isinstance(document, list) or
            not all(isinstance(token, str) for token in document)):
        return None
    if (not idf_document or not isinstance(idf_document, dict) or
            not all(isinstance(token, str) and
                    isinstance(freq, float) for token, freq in idf_document.items())):
        return None
    if not isinstance(k1, float) or not 1.2 <= k1 <= 2 or not isinstance(b, float) or not 0 < b < 1:
        return None
    if (not isinstance(avg_doc_len, float) or avg_doc_len is None
            or not isinstance(doc_len, int) or doc_len is None or isinstance(doc_len, bool)):
        return None

    bm25 = {}
    common_tokens = [vocab, document]
    common_vocab = build_vocabulary(common_tokens)
    for token in common_vocab:
        if token not in idf_document:
            bm25[token] = 0.0
        else:
            token_count = document.count(token)
            bm25[token] = (idf_document[token] * (token_count*(k1 + 1)) /
                           (token_count+k1*(1-b+b*(doc_len/avg_doc_len))))
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
    if (not query or not isinstance(query, str) or
            not stopwords or not isinstance(stopwords, list) or
            not all(isinstance(word, str) for word in stopwords)):
        return None
    if (not indexes or not isinstance(indexes, list) or
            not all(isinstance(doc_index, dict) for doc_index in indexes)):
        return None
    for document in indexes:
        if not all(isinstance(token, str) and
                   isinstance(freq, float) for token, freq in document.items()):
            return None

    query_tokenized = tokenize(query)
    if not query_tokenized:
        return None
    query_tokenized = remove_stopwords(query_tokenized, stopwords)
    if not query_tokenized:
        return None
    doc_index_score = []
    index = 0
    for document in indexes:
        freq_cumulative = 0.0
        for token in query_tokenized:
            if token in document:
                freq_cumulative += document[token]
        doc_index_score.append((index, freq_cumulative))
        index += 1
    return sorted(doc_index_score, key=lambda tup: tup[1], reverse=True)


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
    if (not vocab or not isinstance(vocab, list) or
            not all(isinstance(token, str) for token in vocab)):
        return None
    if (not document or not isinstance(document, list) or
            not all(isinstance(token, str) for token in document)):
        return None
    if (not idf_document or not isinstance(idf_document, dict) or
            not all(isinstance(token, str) and
                    isinstance(freq, float) for token, freq in idf_document.items())):
        return None
    if (not isinstance(avg_doc_len, float) or avg_doc_len is None
            or not isinstance(doc_len, int) or doc_len is None or isinstance(doc_len, bool)):
        return None
    bad_input = (not isinstance(k1, float) or not 1.2 <= k1 <= 2 or not isinstance(b, float) or
                 not 0 < b < 1 or doc_len < 0)
    bad_input = bad_input or not alpha or not isinstance(alpha, float) or alpha < 0
    if bad_input:
        return None

    bm25_with_cutoff = {}
    common_tokens = [vocab, document]
    common_vocab = build_vocabulary(common_tokens)
    for token in common_vocab:
        if token in idf_document and idf_document[token] >= alpha:
            token_count = document.count(token)
            bm25_with_cutoff[token] = (idf_document[token] * (token_count * (k1 + 1)) /
                                       (token_count + k1 * (1 - b + b * (doc_len / avg_doc_len))))
    return bm25_with_cutoff


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not index or not isinstance(index, list) or not file_path or not isinstance(file_path, str):
        return
    with open(file_path, 'w', encoding='utf-8') as file:
        dump(index, file, indent=4)


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
        loaded_index = load(file)
    if isinstance(loaded_index, list):
        return loaded_index
    return None


def calculate_spearman(rank: list[int], golden_rank: list[int]) -> float | None:
    """
    Calculate Spearman's rank correlation coefficient between two rankings.

    Args:
        rank (list[int]): Ranked list ogit  document indices.
        golden_rank (list[int]): Golden ranked list of document indices.

    Returns:
        float | None: Spearman's rank correlation coefficient.

    In case of corrupt input arguments, None is returned.
    """
    if (not rank or not isinstance(rank, list) or
            not all(isinstance(doc_index, int) for doc_index in rank)):
        return None
    if (not golden_rank or not isinstance(golden_rank, list) or
            not all(isinstance(golden_doc_index, int) for golden_doc_index in golden_rank)):
        return None
    observations = len(rank)
    if not len(golden_rank) == observations:
        return None
    rank_difference = 0
    for doc_index in rank:
        if doc_index in golden_rank:
            rank_difference += (rank.index(doc_index) - golden_rank.index(doc_index))**2
    return 1-(6 * rank_difference)/(observations*(observations**2 - 1))
