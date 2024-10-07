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
    if not text or not isinstance(text, str):
        return None

    for symbol in text:
        if not symbol.isalpha():
            text = text.replace(symbol, ' ')
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
    if (not tokens or not isinstance(tokens, list)
            or not all(isinstance(i, str) for i in tokens)):
        return None
    if (not stopwords or not isinstance(stopwords, list)
            or not all(isinstance(i, str) for i in stopwords)):
        return None

    for token in tokens.copy():
        if token in stopwords:
            tokens.remove(token)
    return tokens


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if (not documents or not isinstance(documents, list)
            or not all(isinstance(i, list) for i in documents)
            or not all(isinstance(i, str) for j in documents for i in j)):
        return None

    return sum(documents, [])


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
    if (not vocab or not isinstance(vocab, list)
            or not all(isinstance(i, str) for i in vocab)):
        return None
    if (not document_tokens or not isinstance(document_tokens, list)
            or not all(isinstance(i, str) for i in document_tokens)):
        return None

    freq_dict = {}
    len_document_tokens = len(document_tokens)
    for word in vocab:
        if word not in document_tokens:
            freq_dict[word] = 0.0
            continue
        freq_dict[word] = document_tokens.count(word) / len_document_tokens
    for word in document_tokens:
        if word not in freq_dict:
            freq_dict[word] = document_tokens.count(word) / len_document_tokens
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
    if (not vocab or not isinstance(vocab, list)
            or not all(isinstance(i, str) for i in vocab)):
        return None
    if (not documents or not isinstance(documents, list)
            or not all(isinstance(i, list) for i in documents)
            or not all(isinstance(i, str) for j in documents for i in j)):
        return None

    freq_dict = {}
    len_documents = len(documents)
    for word in vocab:
        if not (word in document for document in documents):
            freq_dict[word] = 0.0
            continue
        num_documents_with_term = sum(1 for document_ in documents if word in document_)
        freq_dict[word] = math.log((len_documents - num_documents_with_term + 0.5)
                                   / (num_documents_with_term + 0.5))
    for document in documents:
        for word in document:
            if word not in freq_dict:
                num_documents_with_term = sum(1 for document_ in documents
                                              if word in document_)
                freq_dict[word] = math.log((len_documents - num_documents_with_term + 0.5)
                                           / (num_documents_with_term + 0.5))
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
    if (not tf or not isinstance(tf, dict)
            or not all((isinstance(key, str) and isinstance(value, float)
                        for key, value in tf.items()))):
        return None
    if (not idf or not isinstance(idf, dict)
            or not all((isinstance(key, str) and isinstance(value, float)
                        for key, value in idf.items()))):
        return None

    tf_idf_dict = {}
    for key in tf:
        if key in idf:
            tf_idf_dict[key] = tf[key] * idf[key]
    if tf_idf_dict:
        return tf_idf_dict
    return None


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
    if (not vocab or not isinstance(vocab, list)
            or not all(isinstance(i, str) for i in vocab)):
        return None
    if (not document or not isinstance(document, list)
            or not all(isinstance(i, str) for i in document)):
        return None
    if (not idf_document or not isinstance(idf_document, dict)
            or not all((isinstance(key, str) and isinstance(value, float)
                        for key, value in idf_document.items()))
            or all(isinstance(i, dict) for i in idf_document.values())):
        return None
    if not k1 or not isinstance(k1, float) or not 1.2 <= k1 <= 2.0:
        return None
    if not b or not isinstance(b, float) or not 0 <= b <= 1:
        return None
    if not avg_doc_len or not isinstance(avg_doc_len, float):
        return None
    if not doc_len or not isinstance(doc_len, int) or isinstance(doc_len, bool):
        return None

    bm25_dict = {}
    for word in document:
        if word not in vocab:
            bm25_dict[word] = 0.0
    for word in vocab:
        if word not in bm25_dict:
            num_word_occur = document.count(word)
            bm25_dict[word] = (idf_document[word] * ((num_word_occur * (k1 + 1))
                                                     / (num_word_occur + k1
                                                        * (1 - b + b * (doc_len / avg_doc_len)))))
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
    if (not indexes or not isinstance(indexes, list) or all(isinstance(i, dict) for i in indexes)
            or not all((isinstance(key, str) and isinstance(value, float)
                        for i in indexes for key, value in i.items()))):
        return None
    if not query or not isinstance(query, str):
        return None
    if (not stopwords or not isinstance(stopwords, list)
            or not all(isinstance(i, str) for i in stopwords)):
        return None


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
    if (not isinstance(vocab, list) or not vocab
            or not isinstance(document, list) or not document):
        return None
    if (not isinstance(avg_doc_len, float) or not avg_doc_len
            or not isinstance(doc_len, int) or not doc_len):
        return None
    if (not isinstance(idf_document, str) or not idf_document
            or not isinstance(alpha, float) or not alpha):
        return None
    if (not all(isinstance(i, str) for i in vocab)
            or not all(isinstance(i, str) for i in document)
            or not all(isinstance(i, (str, float)) for i in idf_document)):
        return None


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if (not isinstance(index, list) or not index
            or not isinstance(file_path, str) or not file_path):
        return None
    if (not all(isinstance(i, dict) for i in index)
            or not all(isinstance(i.items(), str | float) for i in index)):
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
    if not isinstance(rank, list) or not rank or not isinstance(golden_rank, list) or not golden_rank:
        return None
    if not all(isinstance(i, int) for i in rank) or not all(isinstance(i, int) for i in golden_rank):
        return None
