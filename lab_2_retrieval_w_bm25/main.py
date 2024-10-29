"""
Lab 2.

Text retrieval with BM25
"""
import json
## pylint:disable=too-many-arguments, unused-argument
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
    new_text = ''
    for character in text.lower():
        if character.isalpha() is True:
            new_text += character
            continue
        new_text += ' '
    for_tokens = new_text.split(' ')
    tokens = []
    for token in for_tokens:
        if token == '':
            continue
        tokens.append(token)
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
    if not isinstance(tokens, list) or not isinstance(stopwords, list)\
            or not tokens or not stopwords:
        return None
    for i in stopwords:
        if not isinstance(i, str):
            return None
    right_tokens = []
    for i in tokens:
        if not isinstance(i, str):
            return None
        if i not in stopwords:
            right_tokens.append(i)
    return right_tokens


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) or not documents:
        return None
    uni_words = set()
    for tokens in documents:
        if not isinstance(tokens, list):
            return None
        for smth in tokens:
            if not isinstance(smth, str):
                return None
        uni_words.update(tokens)
    return list(uni_words)


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
    if not isinstance(vocab, list) or not isinstance(document_tokens, list)\
            or not vocab or not document_tokens:
        return None
    tf = {}
    length = len(document_tokens)
    for token in document_tokens:
        if not isinstance(token, str):
            return None
        if token not in vocab:
            tf[token] = document_tokens.count(token)/length
    for word in vocab:
        if not isinstance(word, str):
            return None
        if word in document_tokens and word not in tf:
            tf[word] = document_tokens.count(word)/length
            continue
        tf[word] = 0.0
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
    if not isinstance(vocab, list) or not isinstance(documents, list)\
        or not vocab or not documents:
        return None
    for lst in documents:
        if not isinstance(lst, list) or not lst:
            return None
        for token in lst:
            if not isinstance(token,str):
                return None
    idf = {}
    for word in vocab:
        if not isinstance(word, str):
            return None
        n = 0
        for lst in documents:
            if word in lst:
                n += 1
        idf[word] = log((len(documents) + 0.5 - n)/(n + 0.5))
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
    if not isinstance(tf, dict) or not isinstance(idf, dict)\
        or not tf or not idf:
        return None
    for key, value in tf.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
    for key, value in idf.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
    tf_idf = {}
    for key in tf:
        if key not in idf:
            tf_idf[key] = 0.0
        tf_idf[key] = round(tf[key] * idf[key], 4)
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
    if not isinstance(vocab, list) or not isinstance(document, list)\
        or not isinstance(idf_document, dict) or not isinstance(k1, float)\
        or not isinstance(b, float):
        return None
    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int)\
            or not vocab or not document or doc_len is True:
        return None
    for elem in vocab:
        if not isinstance(elem, str) or not idf_document:
            return None
    for elem in document:
        if not isinstance(elem, str):
            return None
    for key, value in idf_document.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
    bm25 = {}
    for word in document:
        if word not in vocab:
            bm25[word] = 0.0
    for key in vocab:
        bm25[key] = (idf_document[key]*(document.count(key)*(k1+1))/
                     (document.count(key)+k1*(1-b+b*(doc_len/avg_doc_len))))
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
    if not isinstance(indexes, list) or not isinstance(query, str) or not indexes or not query:
        return None
    for elem in indexes:
        if not isinstance(elem, dict) or not elem:
            return None
    tokens_query = tokenize(query)
    if not isinstance(tokens_query, list):
        return None
    lst_query = remove_stopwords(tokens_query, stopwords)
    if lst_query is None:
        return None
    rang = []
    for i, dictionary in enumerate(indexes):
        summary = 0.0
        for word in lst_query:
            if word not in dictionary:
                continue
            summary += dictionary[word]
        rang.append((i, round(summary, 4)))
    rang = sorted(rang, key = lambda x: x[1], reverse = True)
    return rang


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
    if not isinstance(vocab, list) or not isinstance(document, list)\
        or not isinstance(idf_document, dict) or not isinstance(k1, float)\
        or not isinstance(b, float):
        return None
    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int)\
            or not vocab or not document or doc_len is True:
        return None
    for elem in vocab:
        if not isinstance(elem, str) or not idf_document:
            return None
    for elem in document:
        if not isinstance(elem, str) or doc_len < 0:
            return None
    for key, value in idf_document.items():
        if not isinstance(key, str) or not isinstance(value, float)\
                or not isinstance(alpha, float):
            return None
    cut_idf = {}
    for key, value in idf_document.items():
        if value < alpha:
            continue
        cut_idf[key] = value
    bm25_cut = {}
    for key in vocab:
        if key not in cut_idf:
            continue
        bm25_cut[key] = (cut_idf[key]*(document.count(key)*(k1+1))/
                     (document.count(key)+k1*(1-b+b*(doc_len/avg_doc_len))))
    return bm25_cut


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not isinstance(index, list) or not isinstance(file_path, str) or not file_path:
        return None
    if '.' not in file_path:
        return None
    with open(file_path, 'w', encoding = 'utf-8') as out_file:
        json.dump(index, out_file)
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
    with open(file_path, 'r', encoding = 'utf-8') as load_file:
        indexes = json.load(load_file)
    if not isinstance(indexes, list):
        return None
    return indexes

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
    if not isinstance(rank, list) or not isinstance(golden_rank, list)\
            or len(rank) != len(golden_rank) or not rank or not golden_rank:
        return None
    n = len(rank)
    sum_square = 0
    for index, value in enumerate(rank):
        if not isinstance(value, int):
            return None
        if value not in golden_rank:
            return 0.0
        sum_square += (index - golden_rank.index(value)) ** 2
    return 1 - (6*sum_square)/(n * (n ** 2 - 1))
