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

    only_letters_text = ''
    for letter in text.lower():
        if letter.isalpha() or letter == ' ':
            only_letters_text += letter
        else:
            only_letters_text += ' '
    tokenized_text = only_letters_text.split()
    if not all(isinstance(symbol, str) for symbol in tokenized_text):
        return None
    return tokenized_text


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
    if (not isinstance(tokens, list)
            or not all(isinstance(token, str) for token in tokens)):
        return None
    if (not isinstance(stopwords, list)
            or not all(isinstance(word, str) for word in stopwords)):
        return None
    if len(tokens) == 0 or len(stopwords) == 0:
        return None

    document_tokens = [word for word in tokens if word not in stopwords]
    if not document_tokens:
        return None
    return document_tokens


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
            or not all(isinstance(list_of_tokens, list) for list_of_tokens in documents)):
        return None
    for list_of_tokens in documents:
        if not all(isinstance(token, str) for token in list_of_tokens):
            return None
    words_from_all_docs_list = list(set(word for sublist in documents for word in sublist))

    if not words_from_all_docs_list:
        return None
    return words_from_all_docs_list


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
    if (not isinstance(document_tokens, list)
            or not all(isinstance(token, str) for token in document_tokens)
            or not document_tokens):
        return None
    if (not isinstance(vocab, list)
            or not all(isinstance(uniq_word, str) for uniq_word in vocab)
            or not vocab):
        return None

    tf_figure_dict = {}
    for word in document_tokens:
        if word not in tf_figure_dict:
            tf_figure_dict[word] = 0.0
        for uniq_word in vocab:
            if uniq_word not in tf_figure_dict:
                tf_figure_dict[uniq_word] = 0.0
            tf_figure_dict[word] = document_tokens.count(word) / len(document_tokens)

    if not tf_figure_dict:
        return None
    return tf_figure_dict


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
    if (not isinstance(vocab, list)
            or not all(isinstance(uniq_word, str) for uniq_word in vocab)
            or not vocab):
        return None
    if (not isinstance(documents, list)
            or not all(isinstance(one_doc_text, list) for one_doc_text in documents)
            or not documents):
        return None
    for one_doc_text in documents:
        if (not all(isinstance(word, str) for word in one_doc_text)
                or not one_doc_text):
            return None

    idf_figure_dict = {}
    for word in vocab:
        for one_doc_text in documents:
            in_how_many_docs_is_met = sum(1 for one_doc_text in documents if word in one_doc_text)
            if documents.index(one_doc_text) == (len(documents) - 1):
                idf_figure_dict[word] \
                    = round(math.log((len(documents) - in_how_many_docs_is_met + 0.5) /
                                     (in_how_many_docs_is_met + 0.5), math.e), 4)

    if not idf_figure_dict:
        return None
    return idf_figure_dict


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
    if (not isinstance(tf, dict) or not tf
        or not all(isinstance(tf_keys, str) for tf_keys in tf.keys()
        or not all(isinstance(tf_values, float) for tf_values in tf.values()))):
        return None
    if (not isinstance(idf, dict) or not idf
        or not all(isinstance(idf_keys, str) for idf_keys in idf.keys()
        or not all(isinstance(idf_values, float) for idf_values in idf.values()))):
        return None

    tf_idf_dict = {}
    for tf_word in tf.keys():
        if tf_word not in idf.keys():
            return None
        tf_idf_result = tf[tf_word] * idf[tf_word]
        tf_idf_dict[tf_word] = tf_idf_result

    if not tf_idf_dict:
        return None
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
    if (not isinstance(vocab, list)
            or not all(isinstance(word_in_vocab, str) for word_in_vocab in vocab)
            or not isinstance(document, list)
            or not all(isinstance(el, str) for el in document)
            or not vocab):
        return None
    if (not isinstance(idf_document, dict)
            or not all(isinstance(key_in_idf_dict, str)
                       for key_in_idf_dict in idf_document.keys())
            or not all(isinstance(value_in_idf_dict, float)
                       for value_in_idf_dict in idf_document.values())
            or not idf_document
            or not document):
        return None
    if (not isinstance(k1, float) or not isinstance(b, float)
            or not isinstance(avg_doc_len, float)
            or not isinstance(doc_len, int)
            or isinstance(doc_len, bool)):
        return None

    bm25_dict = {}

    for word_from_doc in document:
        bm25_dict[word_from_doc] = 0.0

    for word_from_idf in idf_document.keys():
        if word_from_idf not in vocab:
            return None
        if word_from_idf not in bm25_dict:
            bm25_dict[word_from_idf] = 0.0
        freq = document.count(word_from_idf)
        bm25_figure = (idf_document[word_from_idf]
                       * ((freq * (k1 + 1)) / (freq + k1
                                               * (1 - b + b * (doc_len / avg_doc_len)))))
        bm25_dict[word_from_idf] = bm25_figure

    if not bm25_dict:
        return None
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
    if (not isinstance(indexes, list)
        or not all(isinstance(dict_in_indexes, dict) for dict_in_indexes in indexes)
        or not isinstance(stopwords, list)
        or not all(isinstance(word, str) for word in stopwords)
            or not isinstance(query, str)):
        return None
    for dict_in_indexes in indexes:
        for key, value in dict_in_indexes.items():
            if not isinstance(key, str) or not isinstance(value, float):
                return None

    tokenized_query = tokenize(query)
    if not isinstance(tokenized_query, list) or not stopwords:
        return None
    tokenized_query_without_stopwords = remove_stopwords(tokenized_query, stopwords)
    if not isinstance(tokenized_query_without_stopwords, list):
        return None
    l_of_tuples_dict_plus_its_relevance = []

    for dict_index, dict_at_the_index in enumerate(indexes):
        relevance_of_the_particular_doc = 0.0
        for word in tokenized_query_without_stopwords:
            if word in dict_at_the_index:
                relevance_of_the_particular_doc += dict_at_the_index[word]
        l_of_tuples_dict_plus_its_relevance.append((dict_index,
            relevance_of_the_particular_doc))

    l_of_tuples_dict_plus_its_relevance = sorted(l_of_tuples_dict_plus_its_relevance,
                                                 key=lambda x: x[1], reverse=True)

    if not l_of_tuples_dict_plus_its_relevance:
        return None
    return l_of_tuples_dict_plus_its_relevance


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
    if (not isinstance(vocab, list)
            or not all(isinstance(word_in_vocab, str) for word_in_vocab in vocab)
            or not isinstance(document, list)
            or not all(isinstance(el, str) for el in document)
            or not vocab):
        return None
    if (not isinstance(idf_document, dict)
            or not idf_document
            or not isinstance(alpha, float)
            or not isinstance(avg_doc_len, float)
            or not document):
        return None
    for key, value in idf_document.items():
        if not isinstance(key, str) or not isinstance(value, float):
            return None
    if (not isinstance(k1, float) or not isinstance(b, float)
            or not isinstance(doc_len, int)
            or doc_len < 0
            or isinstance(doc_len, bool)):
        return None

    dict_bm25_with_cutoff = {}
    for word_from_doc in document:
        if word_from_doc not in vocab:
            vocab.append(word_from_doc)
            idf_document[word_from_doc] = 0.0

    for word_from_vocab in vocab:
        freq = document.count(word_from_vocab)
        if not isinstance(idf_document.get(word_from_vocab), float):
            return None
        if isinstance(alpha, float):
            if idf_document.get(word_from_vocab) < alpha:
                continue
        bm25_with_cutoff_figure = (idf_document[word_from_vocab]
                                   * ((freq*(k1+1)) / (freq+k1*(1-b+b*(doc_len/avg_doc_len)))))
        dict_bm25_with_cutoff[word_from_vocab] = bm25_with_cutoff_figure

    return dict_bm25_with_cutoff


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if (not isinstance(index, list)
        or not index
        or not all(isinstance(every_dict_in_index, dict)
        for every_dict_in_index in index)
            or not isinstance(file_path, str)):
        return
    for every_dict_in_index in index:
        if not every_dict_in_index:
            return
        for key_in_every_dict_in_index in every_dict_in_index.keys():
            if not isinstance(key_in_every_dict_in_index, str):
                return
        for value_in_every_dict_in_index in every_dict_in_index.values():
            if not isinstance(value_in_every_dict_in_index, float):
                return
    if not isinstance(file_path, str) or not file_path:
        return

    with open(file_path, 'w', encoding="utf-8") as file:
        json.dump(index, file)


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
    with open(file_path, 'r', encoding="utf-8") as file:
        load_index_file = json.load(file)

    return load_index_file


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
    if (not isinstance(rank, list)
            or not all(isinstance(el, int) for el in rank)
            or not isinstance(golden_rank, list)
            or not all(isinstance(g_el, int) for g_el in golden_rank)):
        return None
    if (len(rank) != len(golden_rank)
            or len(rank) == len(golden_rank) == 0):
        return None

    sum_of_squared_differences = 0
    for rank_int in rank:
        if rank_int in golden_rank:
            sum_of_squared_differences += (golden_rank.index(rank_int) - rank.index(rank_int))**2

    return 1 - (6 * sum_of_squared_differences) / (len(rank) * (len(rank)**2 - 1))
