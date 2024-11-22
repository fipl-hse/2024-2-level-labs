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
    text = text.lower()
    for elem in text:
        if not elem.isalpha() and elem != ' ':
            text = text.replace(elem, ' ')
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
    if not isinstance(tokens, list) or not isinstance(stopwords, list):
        return None
    if len(tokens) == 0 or len(stopwords) == 0:
        return None
    for elem in stopwords:
        if not isinstance(elem, str):
            return None
    for elem in tokens:
        if not isinstance(elem, str):
            return None
    tokens_without_stopwords = [elem for elem in tokens if elem not in stopwords]
    return tokens_without_stopwords


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) or len(documents) == 0:
        return None
    if (not all(isinstance(doc, list) for doc in documents)
            or not all(isinstance(token, str) for doc in documents for token in doc)):
        return None
    all_tokens = [token for doc in documents for token in doc]
    return all_tokens


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
    if (not isinstance(vocab, list) or not all(isinstance(token, str)for token in vocab)
            or not vocab):
        return None
    if (not isinstance(document_tokens, list)
            or not all(isinstance(token, str)for token in document_tokens) or not document_tokens):
        return None
    for word in document_tokens:
        if word not in vocab:
            vocab.append(word)
    return {word: document_tokens.count(word)/len(document_tokens) for word in set(vocab)}


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
    if (not isinstance(vocab, list) or not all(isinstance(token, str) for token in vocab)
            or not vocab):
        return None
    if (not isinstance(documents, list) or not all(isinstance(doc, list) for doc in documents)
            or not documents):
        return None
    if not all(isinstance(word, str) for doc in documents for word in doc):
        return None
    idf = {}
    for word in vocab:
        num = 0
        for doc in documents:
            if word in doc:
                num += 1
        idf[word] = math.log((len(documents) - num + 0.5) /
                             (num + 0.5))
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
    if (not isinstance(idf, dict) or not idf
            or not all(isinstance(word, str) for word in idf.keys())
            or not all(isinstance(value, float) for value in idf.values())):
        return None
    if (not isinstance(tf, dict) or not tf
            or not all(isinstance(word, str) for word in tf.keys())
            or not all(isinstance(value, float) for value in tf.values())):
        return None
    tf_idf = {}
    for word, value in tf.items():
        if word not in idf:
            idf[word] = 0.0
        tf_idf[word] = value * idf[word]
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
    if (not isinstance(vocab, list) or not isinstance(document, list)
            or not isinstance(avg_doc_len, float) or not isinstance(doc_len, int)
            or isinstance(doc_len, bool)):
        return None
    if (len(vocab) == 0 or len(document) == 0 or
            not isinstance(vocab[0], str) or not isinstance(document[0], str)):
        return None
    if (not isinstance(idf_document, dict) or not idf_document
            or not isinstance(list(idf_document.values())[0], float)):
        return None
    if not isinstance(k1, float) or not isinstance(b, float):
        return None
    bm25_dict = {}
    for word in document:
        if word in vocab:
            continue
        vocab.append(word)
        idf_document.update({word: 0.0})
    for word in vocab:
        idf = idf_document.get(word)
        if not isinstance(idf, float):
            return None
        bm25 = idf * ((document.count(word) * (k1 + 1)) /
                      (document.count(word) + k1 * (1 - b + b * (doc_len / avg_doc_len))))
        bm25_dict.update({word: bm25})
    return bm25_dict


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
    if (not isinstance(indexes, list) or len(indexes) == 0 or not isinstance(indexes[0], dict)
            or not isinstance(query, str) or not isinstance(stopwords, list)):
        return None
    if (not all(isinstance(story, dict) for story in indexes)
            or not all(isinstance(word, str) for story in indexes for word in story.keys())
            or not all(isinstance(num, float) for story in indexes for num in story.values())):
        return None
    tokens = tokenize(query)
    if not isinstance(tokens, list):
        return None
    correct_query = remove_stopwords(tokens, stopwords)
    if not correct_query:
        return None
    result = []
    for index, story in enumerate(indexes):
        value = sum(story[word] if word in story else 0
                    for word in correct_query)
        result.append((index, value))
    result.sort(key=lambda x: x[1], reverse=True)
    return result


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
    if (not isinstance(vocab, list) or not isinstance(document, list)
            or not isinstance(avg_doc_len, float) or not isinstance(doc_len, int)
            or isinstance(doc_len, bool)):
        return None
    if (len(vocab) == 0 or len(document) == 0 or
            not isinstance(vocab[0], str) or not isinstance(document[0], str))\
            or doc_len < 1:
        return None
    if (not isinstance(idf_document, dict) or not idf_document
            or not isinstance(list(idf_document.values())[0], float)):
        return None
    if not isinstance(k1, float) or not isinstance(b, float) or not isinstance(alpha, float):
        return None
    bm25_dict = {}
    for word in document:
        if word in vocab:
            continue
        vocab.append(word)
        idf_document.update({word: 0.0})
    for word in vocab:
        idf = idf_document.get(word)
        if not isinstance(idf, float):
            return None
        if idf < alpha:
            continue
        bm25 = idf * ((document.count(word) * (k1 + 1)) /
                      (document.count(word) + k1 * (1 - b + b * (doc_len / avg_doc_len))))
        bm25_dict.update({word: bm25})
    return bm25_dict


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not isinstance(index, list) or not isinstance(file_path, str)\
            or len(index) == 0 or len(file_path) == 0:
        return None
    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(index, file)
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
    if not isinstance(file_path, str) or len(file_path) == 0:
        return None
    with open(file_path, "r", encoding="utf-8") as file:
        result = json.load(file)
    if not isinstance(result, list):
        return None
    return result


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
    if not isinstance(rank, list) or not isinstance(golden_rank, list):
        return None
    if len(rank) != len(golden_rank) or len(rank) <= 1:
        return None
    summa = 0.0
    for doc in rank:
        if not isinstance(doc, int):
            return None
        if doc not in golden_rank:
            continue
        summa += ((rank.index(doc) - golden_rank.index(doc)) ** 2)
    spearman = 1 - (6 * summa / (len(rank) * (len(rank) ** 2 - 1)))
    return spearman
