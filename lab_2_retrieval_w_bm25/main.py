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
    out = []
    wrd = ''
    for char in text.lower():
        if char.isalpha():
            wrd += char
        else:
            if wrd != '':
                out.append(wrd)
            wrd = ''
    if not out:
        return None
    return out


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
    if not isinstance(tokens, list):
        return None
    if not isinstance(stopwords, list):
        return None
    srtd = []
    for e in tokens:
        if isinstance(e, str) and e not in stopwords:
            srtd.append(e)
    return srtd


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list):
        return None
    out = []
    for a in documents:
        if not isinstance(a, list):
            return None
        for el in a:
            if not isinstance(el, str):
                return None
            if el not in out:
                out.append(el)
    return out


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
    if not isinstance(vocab, list) or \
            not isinstance(document_tokens, list):
        return None
    if len(document_tokens) == 0:
        return None
    outpt = {}
    for el in vocab:
        if not isinstance(el, str):
            return None
        outpt[el] = document_tokens.count(el) / len(document_tokens)
    return outpt


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
    if not isinstance(vocab, list) or \
            not isinstance(documents, list):
        return None
    ttl = len(documents)
    out = {}
    for el in vocab:
        if not isinstance(el, str):
            return None
        temp = 0
        for a in documents:
            if not (isinstance(a, list)):
                return None
            if a.count(el) > 0:
                temp += 1
        out[el] = math.log1p(ttl / temp)
    return out


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
    if not isinstance(tf, dict) or \
            not isinstance(idf, dict):
        return None
    out = {}
    for el in tf.keys():
        out[el] = tf[el] * idf[el]
    return out


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
    if not isinstance(vocab, list) or \
            not isinstance(document, list) or \
            not isinstance(avg_doc_len, float) or \
            not isinstance(doc_len, int) or \
            not isinstance(idf_document, dict):
        return None
    # if vocab != idf_document.keys:
    # return None
    out = {}
    for el in vocab:
        curidf = idf_document[el]
        curcnt = document.count(el)
        tmpvar = (curcnt * (k1 + 1)) / (curcnt + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        out[el] = curidf * tmpvar
    return out


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
    if not isinstance(indexes, list) or \
            not isinstance(query, str) or \
            not isinstance(stopwords, list):
        return None
    prcsd = remove_stopwords(tokenize(query), stopwords)



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
