"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument

import json
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

    for symbol in text:
        if (symbol.isalpha() or
            symbol == " " or
            symbol == "\n"):
            continue
        text = text.replace(symbol, " ")

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
    if not (isinstance(tokens, list) and
            len(tokens) > 0 and
            all(isinstance(token, str) for token in tokens) and
            isinstance(stopwords, list) and
            len(stopwords) > 0 and
            all(isinstance(word, str) for word in stopwords)):
        return None

    new_tokens = [token for token in tokens if not token in stopwords]
    return new_tokens


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not (isinstance(documents, list) and
            len(documents) > 0 and
            all(isinstance(doc, list) for doc in documents)):
        return None

    unique_words = set()
    for doc in documents:
        if all(isinstance(token, str) for token in doc):
            unique_words = unique_words | set(doc)
        else:
            return None

    return list(unique_words)


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
    if not (isinstance(vocab, list) and
            len(vocab) > 0 and
            all(isinstance(word, str) for word in vocab) and
            isinstance(document_tokens, list) and
            len(document_tokens) > 0 and
            all(isinstance(token, str) for token in document_tokens)):
        return None

    freq_dict = dict.fromkeys(document_tokens + vocab, 0.0)
    for word in document_tokens:
        freq_dict[word] = document_tokens.count(word) / len(document_tokens)

    return freq_dict


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
    if not (isinstance(vocab, list) and
            len(vocab) > 0 and
            all(isinstance(word, str) for word in vocab) and
            isinstance(documents, list) and
            len(documents) > 0 and
            all(isinstance(doc, list) for doc in documents)):
        return None
    for doc in documents:
        if not all(isinstance(word, str) for word in doc):
            return None

    freq_dict = {}
    for word in vocab:
        docs_w_word = len([True for doc in documents if word in doc])
        freq_dict[word] = math.log((len(documents) - docs_w_word + 0.5) / (docs_w_word + 0.5))

    return freq_dict


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
    if not (isinstance(tf, dict) and
            len(tf) > 0 and
            all(isinstance(tf_key, str) and
                isinstance(tf[tf_key], float) for tf_key in tf) and
            isinstance(idf, dict) and
            len(idf) > 0 and
            all(isinstance(idf_key, str) and
                isinstance(idf[idf_key], float) for idf_key in idf)):
        return None

    tf_idf = {}
    for key in tf:
        if key in idf:
            tf_idf[key] = tf[key] * idf[key]
        else:
            return None

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
    if not (isinstance(vocab, list) and
            len(vocab) > 0 and
            isinstance(document, list) and
            len(document) > 0 and
            isinstance(idf_document, dict) and
            len(idf_document) > 0):
        return None
    if not (isinstance(k1, float) and
            isinstance(b, float) and
            isinstance(avg_doc_len, float) and
            avg_doc_len > 0 and
            isinstance(doc_len, int) and
            doc_len is not True):
        return None
    if not (all(isinstance(voc_word, str) for voc_word in vocab) and
            all(isinstance(doc_word, str) for doc_word in document) and
            all(isinstance(idf_key, str) and
                isinstance(idf_document[idf_key], float) for idf_key in idf_document)):
        return None

    result_dict = {}
    for word in set(idf_document) | set(document):
        amount = document.count(word)
        value = 0.0
        if word in idf_document:
            value = (idf_document[word] *
                    (amount * (k1 + 1)) /
                    (amount + k1 * (1 - b + b * doc_len / avg_doc_len)))
        result_dict[word] = value

    return result_dict


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
    if not (isinstance(indexes, list) and
            len(indexes) > 0 and
            isinstance(query, str) and
            len(query) > 0 and
            isinstance(stopwords, list) and
            len(stopwords) > 0):
        return None
    if not (all(isinstance(index, dict) for index in indexes) and
            all(isinstance(word, str) for word in stopwords)):
        return None

    tokenized_query = tokenize(query)
    if not isinstance(tokenized_query, list):
        return None
    query_to_compare = remove_stopwords(tokenized_query, stopwords)
    if not isinstance(query_to_compare, list):
        return None

    doc_sums = []
    for doc_num, doc in enumerate(indexes):
        values_sum = sum(doc[token] for token in query_to_compare if token in doc)
        doc_sums.append((doc_num, values_sum))
    doc_sums.sort(reverse=True, key=lambda a: a[1])
    return doc_sums


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
    if not (isinstance(vocab, list) and
            len(vocab) > 0 and
            isinstance(document, list) and
            len(document) > 0 and
            isinstance(idf_document, dict) and
            len(idf_document) > 0):
        return None
    if not (isinstance(k1, float) and
            isinstance(b, float) and
            isinstance(avg_doc_len, float) and
            avg_doc_len > 0 and
            isinstance(doc_len, int) and
            doc_len is not True):
        return None
    if not (all(isinstance(voc_word, str) for voc_word in vocab) and
            all(isinstance(doc_word, str) for doc_word in document) and
            all(isinstance(idf_key, str) and
                isinstance(idf_document[idf_key], float) for idf_key in idf_document) and
            isinstance(alpha, float)):
        return None

    result_dict = {}
    for word in idf_document:
        if idf_document[word] >= alpha:
            amount = document.count(word)
            value = (idf_document[word] *
                     (amount * (k1 + 1)) /
                     (amount + k1 * (1 - b + b * doc_len / avg_doc_len)))
            result_dict[word] = value

    return result_dict


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not (isinstance(index, list) and
            len(index) > 0 and
            all(isinstance(doc, dict) for doc in index) and
            isinstance(file_path, str) and
            len(file_path) > 0):
        return None

    with open(file_path, "w", encoding="utf-8") as write_file:
        json.dump(index, write_file, indent="\t")
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
    if not (isinstance(file_path, str) and
            len(file_path) > 0):
        return None

    with open(file_path, "r", encoding="utf-8") as read_file:
        val = json.load(read_file)
        if isinstance(val, list):
            return val
    return None


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
    if not (isinstance(rank, list) and
            len(rank) > 0 and
            isinstance(golden_rank, list) and
            len(rank) == len(golden_rank)):
        return None
    if not (all(isinstance(value, int) for value in rank) and
            all(isinstance(gold_val, int) for gold_val in golden_rank)):
        return None

    if not all(val in golden_rank for val in rank):
        return 0.0

    diffs_sum = 0
    pairs = len(rank)
    for rank_index, rank_value in enumerate(rank):
        golden_index = golden_rank.index(rank_value)
        diffs_sum += (rank_index - golden_index)**2
    return 1 - 6 * diffs_sum / (pairs * (pairs**2 - 1))
