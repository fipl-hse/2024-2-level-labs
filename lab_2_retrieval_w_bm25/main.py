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

    if not tokens or not stopwords \
            or not isinstance(tokens, list) or not isinstance(stopwords, list) \
            or not all(isinstance(token, str) for token in tokens) \
            or not all(isinstance(word, str) for word in stopwords) \
            or not all(token for token in tokens) or not all(word for word in stopwords):
        return None

    tokens_prep = []
    for token in tokens:
        if token not in stopwords:
            tokens_prep += [token]
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

    unique_words = []
    for lst_tokens in documents:
        for token in lst_tokens:
            if not token or not isinstance(token, str):
                return None
            if token not in unique_words:
                unique_words += [token]
            else:
                continue
    return unique_words


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
            or not document_tokens or not isinstance(document_tokens, list) \
            or not all(isinstance(un_word, str) for un_word in vocab) \
            or not all(un_word for un_word in vocab) \
            or not all(isinstance(token, str) for token in document_tokens) \
            or not all(token for token in document_tokens):
        return None

    tf_vocab = {}
    for un_word in vocab:
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

    if not vocab or not isinstance(vocab, list) \
            or not documents or not isinstance(documents, list) \
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
            or not idf or not isinstance(idf, dict) \
            or not tf.keys() or not tf.values() \
            or not all(isinstance(key, str) for key in tf.keys()) \
            or not all(isinstance(value, float) for value in tf.values()) \
            or not idf.keys() or not idf.values() \
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
