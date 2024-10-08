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
    tokens = []
    word = []
    only_letters = []
    for elem in text.lower():
        if elem.isalpha():
            word.append(elem)
        else:
            only_letters.append(''.join(word))
            word = []
    for token in only_letters:
        if token:
            tokens.append(token)
    if not isinstance(tokens, list):
        return None
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
    if not isinstance(tokens, list) or not isinstance(stopwords, list):
        return None
    if len(tokens) == 0 or len(stopwords) == 0:
        return None
    for elem in stopwords:
        if not isinstance(elem, str):
            return None
    tokens_without_stopwords = []
    for elem in tokens:
        if not isinstance(elem, str):
            return None
        if elem in stopwords:
            continue
        tokens_without_stopwords.append(elem)
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
    all_tokens = []
    for tokens in documents:
        if not isinstance(tokens, list):
            return None
        for elem in tokens:
            if not isinstance(elem, str):
                return None
        all_tokens.extend(tokens)
    return list(set(all_tokens))


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
            or len(vocab) == 0 or len(document_tokens) == 0:
        return None
    for word in vocab:
        if not isinstance(word, str):
            return None
    for word in document_tokens:
        if not isinstance(word, str):
            return None
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
    if not isinstance(vocab, list) or not isinstance(documents, list)\
            or len(vocab) == 0 or len(documents) == 0:
        return None
    for word in vocab:
        if not isinstance(word, str):
            return None
    idf = {}
    tokens = []
    for doc in documents:
        if not isinstance(doc, list):
            return None
        for word in doc:
            freq = doc.count(word)
            if freq > 1:
                doc.remove(word)
        tokens.extend(doc)
    for word in tokens:
        if not isinstance(word, str):
            return None
        idf_word = math.log((len(documents) - tokens.count(word) + 0.5) / (tokens.count(word) + 0.5))
        idf.update({word: round(idf_word, 4)})
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
    if not isinstance(tf, dict) or not isinstance(idf, dict):
        return None
    if len(tf) == 0 or len(idf) == 0:
        return None
    tf_idf = {}
    for word, value in tf.items():
        value_idf = idf.get(word)
        if not isinstance(word, str) or not isinstance(value, float) or not isinstance(value_idf, float):
            return None
        tf_idf_word = value * value_idf
        tf_idf.update({word: tf_idf_word})
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
    if (not isinstance(vocab, list) or not isinstance(document, list) or len(vocab) == 0 or len(document) == 0
            or not isinstance(vocab[0], str) or not isinstance(document[0], str)):
        return None
    if not isinstance(idf_document, dict) or not idf_document or not isinstance(list(idf_document.values())[0], float):
        return None
    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) or isinstance(doc_len, bool):
        return None
    if not isinstance(k1, float) or not isinstance(b, float):
        return None
    bm25_dict = {}
    for word in document:
        if word not in vocab:
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
    tokens = tokenize(query)
    if not isinstance(tokens, list):
        return None
    correct_query = remove_stopwords(tokens, stopwords)
    if not correct_query:
        return None
    result = []
    for index, story in enumerate(indexes):
        value = 0.0
        for word in correct_query:
            if word not in story.keys():
                continue
            num = story.get(word)
            if not isinstance(num, float):
                return None
            value += num
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
