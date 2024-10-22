"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument
import re
import math
import json


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
    lower = text.lower()
    list_for_words = re.findall('[a-zа-я]+', lower)
    return list_for_words


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
    if tokens is None or stopwords is None:
        return None
    if (not isinstance(tokens, list) or not isinstance(stopwords, list) or
            not all(isinstance(i, str) for i in tokens) or
            not all(isinstance(k, str) for k in stopwords)):
        return None
    if not tokens or not stopwords:
        return None
    without_stop = [word for word in tokens if word not in stopwords]
    return without_stop


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not documents:
        return None
    if not isinstance(documents, list):
        return None
    if not all(isinstance(item, list) and all(isinstance(elem, str)
                                              for elem in item) for item in documents):
        return None
    main_list = set()
    for words in documents:
        main_list.update(words)
    return list(main_list)


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
    if not vocab or not document_tokens:
        return None
    if vocab is None or document_tokens is None:
        return None
    if (not isinstance(vocab, list) or not isinstance(document_tokens, list) or
            not all(isinstance(i, str) for i in document_tokens) or
            not all(isinstance(k, str) for k in vocab)):
        return None
    dictionary_for_tf = {}
    for word in vocab:
        dictionary_for_tf[word] = 0.0
        if word in document_tokens:
            dictionary_for_tf[word] = document_tokens.count(word) / len(document_tokens)
    for token in document_tokens:
        if token in dictionary_for_tf:
            continue
        dictionary_for_tf[token] = document_tokens.count(token) / len(document_tokens)
    return dictionary_for_tf


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
    if not vocab or not documents:
        return None
    if (not isinstance(vocab, list) or not isinstance(documents, list) or
            not all(isinstance(p, str) for p in vocab) or
            not all(isinstance(k, list) and all(isinstance(elem, str)
                                                for elem in k) for k in documents)):
        return None
    counter_for_documents = len(documents)
    dictionary_for_idf = {}
    for word in vocab:
        dictionary_for_idf[word] = 0.0
        counter = 0.0
        for document in documents:
            if word in document:
                counter += 1.0
            value = (counter_for_documents - counter + 0.5)/(counter + 0.5)
            dictionary_for_idf[word] = math.log(value)
    return dictionary_for_idf


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
    if not tf or not idf:
        return None
    if not isinstance(tf, dict) or not isinstance(idf, dict):
        return None
    if (not all(isinstance(key, str) for key in tf.keys()) or
            not all(isinstance(value, float) for value in tf.values())):
        return None
    if (not all(isinstance(key, str) for key in idf.keys()) or
            not all(isinstance(value, float) for value in idf.values())):
        return None
    dictionary_for_result = {}
    for key, value in tf.items():
        dictionary_for_result[key] = 0.0
        if key in idf.keys():
            dictionary_for_result[key] = tf[key] * idf[key]

    if not dictionary_for_result:
        return None

    return dictionary_for_result


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
    if ((not isinstance(vocab, list) or not isinstance(document, list) or
            not isinstance(idf_document, dict))):
        return None
    if ((not all(isinstance(value, str) for value in vocab)) or
            (not all(isinstance(value, str) for value in document)) or
            (not all(isinstance(key, str) and isinstance(value, float) for
                     key, value in idf_document.items()))):
        return None
    if ((not isinstance(k1, float)) or (not isinstance(b, float)) or
            ((not isinstance(avg_doc_len, float)) or (not isinstance(doc_len, int)) or
            (isinstance(doc_len, bool)))):
        return None
    bm25 = {}

    for word_in_doc in document:
        bm25[word_in_doc] = 0.0

    for word_in_idf in idf_document.keys():
        if word_in_idf not in vocab:
            return None
        if word_in_idf not in bm25:
            bm25[word_in_idf] = 0.0
        bm25[word_in_idf] = (idf_document[word_in_idf] * document.count(word_in_idf) *
                                           (k1 + 1) / (document.count(word_in_idf) + k1 *
                                                       (1 - b + b * doc_len / avg_doc_len)))
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
    if ((not isinstance(query, str)) or (not isinstance(stopwords, list)) or
            (not all(isinstance(i, str) for i in stopwords)) or (not isinstance(indexes, list))
            or (not all(isinstance(k, dict) for k in indexes))):
        return None
    for index in indexes:
        for key, value in index.items():
            if not isinstance(key, str) or not isinstance(value, float):
                return None

    letters = tokenize(query)
    if not isinstance(letters, list):
        return None
    without_stopwords = remove_stopwords(letters, stopwords)
    if not isinstance(without_stopwords, list):
        return None

    list_with_index = []
    for index_from_indexes, metrica in enumerate(indexes):
        value_of_whole_document = 0.0
        for word in without_stopwords:
            if word in metrica:
                value_of_whole_document += metrica[word]
        tuple_of_metrica = (index_from_indexes, value_of_whole_document)
        list_with_index.append(tuple_of_metrica)

    pairs = sorted(list_with_index, key=lambda pair: pair[1], reverse=True)
    if not list_with_index:
        return None
    return pairs


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
    # if not vocab or not document or not idf_document:
    #     return None
    if ((not isinstance(vocab, list)) or
            (not isinstance(document, list)) or (not isinstance(idf_document, dict)) or
            not vocab or not document):
        return None
    if ((not all(isinstance(value, str) for value in vocab)) or
            (not all(isinstance(value, str) for value in document)) or
            (not all(isinstance(key, str) and isinstance(value, float) for
                     key, value in idf_document.items()))):
        return None
    if ((not isinstance(k1, float)) or
            (not isinstance(b, float)) or (not isinstance(alpha, float)) or
            not isinstance(avg_doc_len, float) or not idf_document):
        return None
    if (not isinstance(doc_len, int) or isinstance(doc_len, bool)
            or doc_len < 0):
        return None
    bm25 = {}
    for word_in_idf, score in idf_document.items():
        if word_in_idf not in vocab:
            return None
        if word_in_idf not in bm25 and score > alpha:
            bm25[word_in_idf] = 0.0
        if score > alpha:
            bm25[word_in_idf] = (score * document.count(word_in_idf) * (k1 + 1) /
                                 (document.count(word_in_idf) + k1 *
                                  (1 - b + b * doc_len / avg_doc_len)))
    return bm25


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if (not isinstance(index, list) or (not all(isinstance(item, dict)
        and all(isinstance(k, str) and
                isinstance(v, float) for k, v in item.items())
                for item in index) or (not isinstance(file_path, str)))):
        return None
    if not index or not file_path:
        return None
    with open(file_path, "w", encoding='utf-8') as write_file:
        json.dump(index, write_file, indent=4)
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
    with open(file_path, 'r', encoding='utf-8') as file:
        file_for_query: list[dict[str, float]] = json.load(file)
    if not isinstance(file_for_query, list):
        return None
    return file_for_query


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
    if (not all(isinstance(i, int) for i in rank) or
            not all(isinstance(k, int) for k in golden_rank)):
        return None
    if len(rank) != len(golden_rank):
        return None
    length = len(rank)
    differences = 0

    for ind in rank:
        if ind in golden_rank:
            differences += (rank.index(ind) - golden_rank.index(ind)) ** 2
    return 1 - (6 * differences) / (length * (length ** 2 - 1))
