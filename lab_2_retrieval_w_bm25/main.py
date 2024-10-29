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
    tokens = []
    for token in text.lower().split():
        clean_word = ''.join(char if char.isalpha() else ' ' for char in token)
        for word in clean_word.split():
            if word:
                tokens.append(word)
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

    if not (isinstance(tokens, list) or all(isinstance(token, str)
                                               for token in tokens) or tokens):
        return None
    if not (isinstance(stopwords, list) or all(isinstance(word, str)
                                                  for word in stopwords) or stopwords):
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

    if not (isinstance(documents, list) or all(isinstance(doc, list) for doc in documents)):
        return None
    if not all(isinstance(token, str) for doc in documents for token in doc):
        return None
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc)
    return list(vocabulary) or None


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

    if not (vocab or document_tokens):
        return None
    if vocab is None or document_tokens is None:
        return None
    if not (isinstance(vocab, list) or isinstance(document_tokens, list)):
        return None
    if not all(isinstance(word, str) for word in vocab + document_tokens):
        return None
    unique_words = set(vocab).union(set(document_tokens))
    tf_dict = {word: 0 for word in unique_words}
    if not document_tokens:
        return {word: 0.0 for word in vocab}
    for word in document_tokens:
        if word in tf_dict:
            tf_dict[word] += 1
    doc_length = len(document_tokens)
    return {word: (count / doc_length) if doc_length > 0 else 0.0 for
            word, count in tf_dict.items()}


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

    if not (vocab or documents):
        return None
    if not (isinstance(vocab, list) or isinstance(documents, list)):
        return None
    if not all(isinstance(word, str) for word in vocab):
        return None
    if not all(isinstance(doc, list) and
               all(isinstance(token, str) for token in doc) for doc in documents):
        return None
    length = len(documents)
    idf_dict = {}
    for word in vocab:
        count = 0
        for doc in documents:
            if word in doc:
                count += 1
        idf_dict[word] = math.log((length - count + 0.5) / (count + 0.5))
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

    if not (tf or idf):
        return None
    if not (isinstance(tf, dict) or isinstance(idf, dict)):
        return None
    if (not (all(isinstance(word, str) for word in tf.keys())
             or all(isinstance(word, str) for word in idf.keys()))):
        return None
    if not (all(isinstance(value, (int, float)) for value in tf.values()) or all(
            isinstance(value, (int, float)) for value in idf.values())):
        return None
    tf_idf_dict = {word: tf[word] * idf.get(word, 0) for word in tf.keys()}
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

    if not vocab or not document or not idf_document:
        return None
    if (not isinstance(vocab, list) or not isinstance(document, list) or
            not isinstance(idf_document, dict)):
        return None
    if ((not all(isinstance(value, str) for value in vocab)) or
            (not all(isinstance(value, str) for value in document))
            or (not all(isinstance(key, str) and isinstance(value, float) for
                        key, value in idf_document.items()))):
        return None
    if ((not isinstance(k1, float)) or (not isinstance(b, float))
            or ((not isinstance(avg_doc_len, float)) or
                (not isinstance(doc_len, int)) or (isinstance(doc_len, bool)))):
        return None
    if avg_doc_len is None or avg_doc_len <= 0:
        return None
    bm25 = {}
    for token in vocab:
        if token not in idf_document:
            bm25[token] = 0.0
        else:
            token_count = document.count(token)
            denominator = token_count + k1 * (1 - b + b * (doc_len / avg_doc_len))
            if denominator == 0:
                bm25[token] = 0.0
            else:
                bm25[token] = idf_document[token] * (token_count * (k1 + 1)) / denominator
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

    if (not isinstance(indexes, list) or
            not all(isinstance(el, dict) and
                    all(isinstance(key, str)
                        and isinstance(value, float)
                        for key, value in el.items()) for el in indexes)):
        return None
    if (not isinstance(stopwords, list) or not all(isinstance(el, str) for el in stopwords)
            or not isinstance(query, str)):
        return None
    if not (indexes or query):
        return None
    query_tokenized = tokenize(query)
    if not query_tokenized:
        return None
    query_tokenized = remove_stopwords(query_tokenized, stopwords)
    if not query_tokenized:
        return None
    doc_scores = []
    index = 0
    for document in indexes:
        freq = 0.0
        for token in query_tokenized:
            if token in document:
                freq += document[token]
        doc_scores.append((index, freq))
        index += 1
    return sorted(doc_scores, key=lambda tup: tup[1], reverse=True)


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
