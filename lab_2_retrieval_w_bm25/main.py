"""
Lab 2.

Text retrieval with BM25
"""
import math

# pylint:disable=too-many-arguments, unused-argument


def tokenize(text: str) -> list[str] | None:
    """
    Tokenize the input text into lowercase words without punctuation, digits and other symbols.

    Args:
        text (str): The input text to tokenize.

    Returns:
        list[str] | None: A list of words from the text.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(text, str) or len(text) < 1:
        return None

    for symbol in text:
        if symbol.isalpha() or symbol == " ":
            continue
        text = text.replace(f"{symbol}", " ")

    new_text = text.split(" ")
    result = []
    for word in new_text:
        if not word.isalpha():
            continue
        result.append(word.lower())

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
    if tokens is None or not isinstance(tokens, list) or not isinstance(stopwords, list):
        return None
    for token in tokens:
        if not isinstance(token, str):
            return None
    for word in stopwords:
        if not isinstance(word, str):
            return None
    if len(tokens) == 0 or len(stopwords) == 0:
        return None

    result = [token for token in tokens if token not in stopwords]
    return result


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

    vocab = []

    for document in documents:
        if not isinstance(document, list) or len(document) == 0:
            return None
        for word in document:
            if not isinstance(word, str) or len(word) == 0:
                return None
            if word not in vocab:
                vocab.append(word)

    if not vocab:
        return None
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
    if (not isinstance(vocab, list) or not isinstance(document_tokens, list)
            or len(vocab) == 0 or len(document_tokens) == 0):
        return None
    if (not all(isinstance(v, str) for v in vocab)
            or not all(isinstance(d, str) for d in document_tokens)):
        return None

    len_document = len(document_tokens)
    token_set = set(set(vocab) | set(document_tokens))
    tf_dict = {}
    for token in token_set:
        tf = document_tokens.count(token) / len_document
        tf_dict[token] = tf

    return tf_dict


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
    if (not isinstance(vocab, list) or not isinstance(documents, list)
            or len(vocab) == 0 or len(documents) == 0):
        return None
    if (not all(isinstance(token, str) for token in vocab) or
            not all(token.isalpha() for token in vocab) or
            not all(isinstance(document, list) for document in documents)):
        return None
    for document in documents:
        if (not all(isinstance(token, str) for token in document) or
                not all(token.isalpha() for token in document)):
            return None
    idf_dict = {}

    for token in vocab:
        token_count = 0
        for document in documents:
            if token in document:
                token_count += 1

        idf = math.log((len(documents) - token_count + 0.5) / (token_count + 0.5))
        idf_dict[token] = idf

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
    if (not isinstance(tf, dict) or not isinstance(idf, dict) or
            not all(isinstance(key, str) for key in tf.keys()) or
            not all(isinstance(value, float) for value in tf.values())):
        return None

    return {token: tf[token] * idf[token] for token in tf if token in idf} or None


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
    bad_input = (not isinstance(vocab, list)
                 or not all(isinstance(token, str) for token in vocab)
                 or not isinstance(document, list)
                 or not all(isinstance(token, str) for token in document)
                 or len(vocab) == 0 or len(document) == 0
                 or not isinstance(idf_document, dict)
                 or len(idf_document) == 0
                 or not all(isinstance(key, str) for key in idf_document.keys())
                 or not all(isinstance(value, float) for value in idf_document.values())
                 or not isinstance(k1, float) or not isinstance(b, float)
                 or not isinstance(avg_doc_len, float) or avg_doc_len is None
                 or not isinstance(doc_len, int) or doc_len is None
                 or isinstance(doc_len, bool))

    if bad_input:
        return None

    bm25_dict = {}
    all_token_list = list(set(vocab) | set(document))
    for token in all_token_list:
        token_count = 0
        if token in document:
            token_count += document.count(token)
        if token in idf_document.keys():
            bm25_dict[token] = (idf_document[token] *
                                ((token_count * (k1 + 1)) /
                                 (token_count + k1 * (1 - b + b *
                                                      (doc_len / avg_doc_len)))))
        else:
            bm25_dict[token] = 0

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
    bad_input = (not isinstance(indexes, list)
                 or not all(isinstance(index, dict) for index in indexes)
                 or not all(isinstance(key, str) and isinstance(value, float)
                            for index in indexes for key, value in index.items())
                 or indexes is None or indexes == []
                 or not isinstance(query, str) or isinstance(query, bool)
                 or query is None or query == ''
                 or not isinstance(stopwords, list) or
                 stopwords is None or stopwords == []
                 or not all(isinstance(word, str) for word in stopwords))
    if bad_input:
        return None

    tokenized_query = tokenize(query)
    if tokenized_query is None:
        return None
    processed_query = remove_stopwords(tokenized_query, stopwords)
    if processed_query is None:
        return None

    result_not_sorted = {}

    for bm25_tfidf_dict in indexes:
        value_sum = 0.0
        i = indexes.index(bm25_tfidf_dict)
        for token in processed_query:
            if token in bm25_tfidf_dict:
                value_sum += bm25_tfidf_dict[token]
        result_not_sorted[i] = value_sum

    result_lst = list(result_not_sorted.items())
    result_sorted = sorted(result_lst, key=lambda x: x[1], reverse=True)
    return result_sorted


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
