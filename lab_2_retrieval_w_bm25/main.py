"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument
import math
import re


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
    return re.sub(r'[^\s\w]+|\d+', r' ', text.lower()).split()


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
    if (not tokens or not stopwords or not isinstance(tokens, list) or
            not isinstance(stopwords, list)):
        return None
    if (not all(isinstance(token, str) for token in tokens) or
            not all(isinstance(stopword, str) for stopword in stopwords)):
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
    if (not isinstance(documents, list) or not documents or
            not all(isinstance(document, list) for document in documents)):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None
    vocab = []
    for document in documents:
        for term in document:
            if term not in vocab:
                vocab.append(term)
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
    if (not vocab or not document_tokens or not isinstance(vocab, list) or
            not isinstance(document_tokens, list)):
        return None
    if (not all(isinstance(term, str) for term in vocab) or
            not all(isinstance(token, str) for token in document_tokens)):
        return None
    for token in document_tokens:
        if token not in vocab:
            vocab.append(token)
    return dict(zip(vocab, [document_tokens.count(term) / len(document_tokens) for term in vocab]))


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
    if not documents or not vocab or not isinstance(documents, list) or not isinstance(vocab, list):
        return None
    if (not all(isinstance(document, list) for document in documents) or
            not all(isinstance(term, str) for term in vocab)):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None
    idf_dict = {}
    for term in vocab:
        documents_w_term = 0
        for document in documents:
            if term in document:
                documents_w_term += 1
        idf_dict[term] = math.log((len(documents) - documents_w_term + 0.5)
                                  / (documents_w_term + 0.5))
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
    if (not tf or not isinstance(tf, dict) or not idf or not isinstance(idf, dict) or
            not all(isinstance(term, str) for term in idf)):
        return None
    return {term: tf[term] * idf[term] for term in tf if term in idf} or None


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
    if (not vocab or not document or not idf_document or not isinstance(vocab, list) or
            not isinstance(document, list) or not isinstance(idf_document, dict) or
            not isinstance(doc_len, int) or not isinstance(k1, float)):
        return None
    if (not isinstance(b, float) or not isinstance(avg_doc_len, float) or
            isinstance(doc_len, bool) or not all(isinstance(term, str) for term in vocab)):
        return None
    if not all(isinstance(term, str) for term in idf_document):
        return None
    if (not all(isinstance(term, str) for term in document) or
            not all(isinstance(freq, float) for freq in idf_document.values())):
        return None
    vocab.extend([term for term in document if term not in vocab])
    idf_document.update({term: 0.0 for term in vocab if term not in idf_document})
    return {term: (idf_document[term] * ((document.count(term) * (k1 + 1))
                                         / (document.count(term)
                                            + k1 * (1 - b + b * (doc_len / avg_doc_len)))))
            for term in vocab}


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
    if (not indexes or not query or not stopwords or not isinstance(indexes, list) or
            not isinstance(stopwords, list) or not isinstance(query, str) or
            not all(isinstance(word, str) for word in stopwords) or
            not all(isinstance(index, dict) for index in indexes)):
        return None
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return None
    clean_query = remove_stopwords(tokenized_query, stopwords)
    if not clean_query:
        return None
    index_score_list = []
    for index in indexes:
        metric_score = 0.0
        index.update({term: 0.0 for term in clean_query if term not in index})
        for term in clean_query:
            metric_score += index[term]
        index_score = (indexes.index(index), metric_score)
        index_score_list.append(index_score)
    return sorted(index_score_list, key=lambda tple: tple[-1], reverse=True)


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
