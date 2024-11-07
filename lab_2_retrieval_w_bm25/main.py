"""
Lab 2.

Text retrieval with BM25
"""

from math import log

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

    if not text or not isinstance(text, str):
        return None

    text_prep = []
    for symb in text.lower():
        if symb.isalpha():
            text_prep.append(symb)
        else:
            text_prep.append(' ')
    return ''.join(text_prep).split()


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

    if not tokens or not isinstance(tokens, list) \
            or not all(isinstance(token, str) for token in tokens) \
            or not all(token for token in tokens):
        return None
    if not stopwords or not isinstance(stopwords, list) \
            or not all(isinstance(word, str) for word in stopwords) \
            or not all(word for word in stopwords):
        return None

    tokens_prep = []
    for token in tokens:
        if token not in stopwords:
            tokens_prep.append(token)
    return tokens_prep


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """

    if not documents or not isinstance(documents, list) \
            or not all(isinstance(lst_tokens, list) for lst_tokens in documents) \
            or not all(lst_tokens for lst_tokens in documents):
        return None

    unique_words = set()
    for lst_tokens in documents:
        for token in lst_tokens:
            if not token or not isinstance(token, str):
                return None
        unique_words |= set(lst_tokens)
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

    if not vocab or not isinstance(vocab, list) \
            or not all(isinstance(un_word, str) for un_word in vocab) \
            or not all(un_word for un_word in vocab):
        return None
    if not document_tokens or not isinstance(document_tokens, list) \
            or not all(isinstance(token, str) for token in document_tokens) \
            or not all(token for token in document_tokens):
        return None

    tf_vocab = {}
    for un_word in set(vocab) | set(document_tokens):
        tf_vocab[un_word] = document_tokens.count(un_word)/len(document_tokens)
    return tf_vocab


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

    if not vocab or not isinstance(vocab, list):
        return None
    if not documents or not isinstance(documents, list) \
            or not all(isinstance(tok_doc, list) for tok_doc in documents) \
            or not all(tok_doc for tok_doc in documents):
        return None

    idf_vocab = {}
    for un_word in vocab:
        if not un_word or not isinstance(un_word, str):
            return None
        count = 0
        for doc in documents:
            for i in doc:
                if not i or not isinstance(i, str):
                    return None
            if un_word in doc:
                count += 1
            else:
                continue
        idf_vocab[un_word] = log((len(documents)-count+0.5)/(count+0.5))
    return idf_vocab


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

    if not tf or not isinstance(tf, dict) \
            or not idf or not isinstance(idf, dict):
        return None
    if not tf.keys() or not tf.values() \
            or not all(isinstance(key, str) for key in tf.keys()) \
            or not all(isinstance(value, float) for value in tf.values()):
        return None
    if not idf.keys() or not idf.values() \
            or not all(isinstance(key, str) for key in idf.keys()) \
            or not all(isinstance(value, float) for value in idf.values()):
        return None

    tf_idf_vocab = {}
    for token in tf:
        tf_idf_vocab[token] = tf[token] * idf[token]
    return tf_idf_vocab


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

    if not vocab or not k1 or not b \
            or not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
        return None
    if not document or not isinstance(document, list) \
            or not all(isinstance(token, str) for token in document):
        return None
    if not idf_document or not isinstance(idf_document, dict) \
            or not all(isinstance(key, str) for key in idf_document) \
            or not all(isinstance(value, float) for value in idf_document.values()):
        return None
    if not isinstance(k1, float) or not isinstance(b, float) \
            or not 1.2 <= k1 <= 2.0 or not 0 <= b <= 1:
        return None
    if not avg_doc_len or not doc_len \
            or not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) \
            or isinstance(doc_len, bool):
        return None

    bm25_vocab = {}
    for word in set(vocab) | set(document):
        if word in idf_document:
            word_freq = document.count(word)
            bm25_vocab[word] = idf_document[word] * ((word_freq * (k1 + 1)) /
                                        (word_freq + k1 * (1 - b + b * (doc_len / avg_doc_len))))
        else:
            bm25_vocab[word] = 0.0
    return bm25_vocab


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

    if not indexes or not isinstance(indexes, list) \
            or not all(isinstance(index_dict, dict) for index_dict in indexes) \
            or not all(isinstance(key, str) for item in indexes for key in item) \
            or not all(isinstance(value, float) for item in indexes for value in item.values()):
        return None
    if not query or not stopwords \
            or not isinstance(query, str) or not isinstance(stopwords, list) \
            or not all(isinstance(word, str) for word in stopwords):
        return None
    if not tokenize(query) or not isinstance(tokenize(query), list) \
            or not all(isinstance(item, str) for item in tokenize(query)) \
            or not remove_stopwords(tokenize(query), stopwords):
        return None

    tokenized_query = tokenize(query)
    if tokenized_query is None:
        return None
    prep_query = remove_stopwords(tokenized_query, stopwords)
    if prep_query is None:
        return None
    answer = []
    for doc_indexes in indexes:
        index = 0.0
        for word in prep_query:
            if word in doc_indexes:
                index += doc_indexes[word]
        answer.append((indexes.index(doc_indexes), index))
    answer = sorted(answer, key=lambda x: x[1], reverse=True)
    return answer


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
