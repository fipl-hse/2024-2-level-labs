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
    result = []
    word = ''
    for token in text.lower():
        if not token.isalpha():
            if word:
                result.append(word)
            word = ''
            continue
        word = ''.join([word, token])
    return result


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
    if (not stopwords or not isinstance(stopwords, list)
            or not all(isinstance(stopword, str) for stopword in stopwords)):
        return None
    for i in tokens[:]:
        if i in stopwords:
            tokens.remove(i)
    return tokens or None


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if (not isinstance(documents, list)
            or not all(isinstance(document, list) for document in documents)):
        return None
    for document in documents:
        if not isinstance(document, list):
            return None
        for word in document:
            if not isinstance(word, str):
                return None
    vocab = set(sum(documents, []))
    return list(vocab) or None


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
    if not vocab or not isinstance(vocab, list) or not isinstance(document_tokens, list):
        return None
    if (not all(isinstance(i, str) for i in vocab)
            or not all(isinstance(i, str) for i in document_tokens)):
        return None
    if len(document_tokens) == 0:
        return None
    unique_words = list(set(vocab + document_tokens))
    tf_dict = {token: document_tokens.count(token) / len(document_tokens) for token in unique_words}
    return tf_dict or None


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
    idf_dict = {}
    for word in vocab:
        word_count = 0
        for document in documents:
            if word in document:
                word_count += 1
        x = (len(documents) - word_count + 0.5) / (word_count + 0.5)
        idf = math.log(x)
        idf_dict[word] = idf
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
    if not tf or not idf or not isinstance(tf, dict) or not isinstance(idf, dict):
        return None
    if not all(isinstance(word, str) for word in idf):
        return None
    tf_idf_dict = {}
    for word in tf:
        if word not in idf:
            return None
        tf_idf_dict[word] = tf[word] * idf[word]
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
    if (not isinstance(vocab, list) or not isinstance(document, list)
            or not isinstance(idf_document, dict)):
        return None
    if (not all(isinstance(word, str) for word in vocab)
            or not all(isinstance(word, str) for word in document)):
        return None
    if not all(isinstance(key, str)
               and isinstance(value, float) for key, value in idf_document.items()):
        return None
    if (not isinstance(k1, float) or not isinstance(b, float)
            or not isinstance(avg_doc_len, float) or not isinstance(doc_len, int)
            or isinstance(doc_len, bool)):
        return None
    bm_dict = {}
    unique_words = list(set(vocab + document))
    for token in unique_words:
        numerator = document.count(token) * (k1 + 1)
        denominator = document.count(token) + k1 * (1 - b + b * doc_len/avg_doc_len)
        if token not in idf_document:
            bm_dict[token] = 0.0
        else:
            bm_dict[token] = idf_document[token] * numerator / denominator
    return bm_dict or None


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
    if (not isinstance(indexes, list) or not isinstance(query, str)
            or not isinstance(stopwords, list)):
        return None
    if not all(isinstance(ind, dict) for ind in indexes):
        return None
    query_tokens = tokenize(query)
    if not query_tokens:
        return None
    query_words = remove_stopwords(query_tokens, stopwords)
    if not query_words:
        return None
    res = []
    for ind, dictionary in enumerate(indexes):
        num = 0.0
        for word in query_words:
            if word in dictionary:
                num += dictionary[word]
        res.append((ind, num))
    res.sort(key=lambda x: x[-1], reverse=True)
    return res or None


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
    if (not vocab or not document or not idf_document
            or not isinstance(document, list) or not isinstance(idf_document, dict)):
        return None
    if (not isinstance(vocab, list) or not all(isinstance(word, str) for word in vocab)
            or not all(isinstance(word, str) for word in document)):
        return None
    if not all(isinstance(key, str)
               and isinstance(value, float) for key, value in idf_document.items()):
        return None
    if not isinstance(k1, float) or not isinstance(b, float) or not isinstance(avg_doc_len, float):
        return None
    if (isinstance(doc_len, bool) or not isinstance(doc_len, int)
            or doc_len < 0 or not isinstance(alpha, float)):
        return None
    bm_dict_w_cutoff = {}
    unique_words = list(set(vocab + document))
    for token in unique_words:
        if token not in idf_document or idf_document[token] < alpha:
            continue
        numerator = document.count(token) * (k1 + 1)
        denominator = document.count(token) + k1 * (1 - b + b * abs(doc_len) / avg_doc_len)
        bm_dict_w_cutoff[token] = idf_document[token] * numerator / denominator
    return bm_dict_w_cutoff


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not isinstance(index, list) or not isinstance(file_path, str) or not file_path or not index:
        return None
    for dictionary in index:
        if not isinstance(dictionary, dict):
            return None
        if not all(isinstance(key, str)
                   and isinstance(value, float) for key, value in dictionary.items()):
            return None
    with open(file_path, 'w', encoding='utf-8') as out_file:
        json.dump(index, out_file, indent=4)
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
    if not isinstance(file_path, str) or not file_path:
        return None
    with open(file_path, 'r', encoding='utf-8') as dict_file:
        profile = json.load(dict_file)
    return profile if isinstance(profile, list) else None


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
    if (not rank or not golden_rank or not isinstance(rank, list)
            or not isinstance(golden_rank, list)):
        return None
    if len(rank) != len(golden_rank):
        return None
    if (not all(isinstance(num, int) for num in rank)
            or not all(isinstance(num, int) for num in golden_rank)):
        return None
    summ = 0
    n = len(rank)
    for ind, num in enumerate(rank):
        if num not in golden_rank:
            return 0.0
        summ += (ind - golden_rank.index(num)) ** 2
    spearman = 1 - (6 * summ) / (n * (n ** 2 - 1))
    return spearman
