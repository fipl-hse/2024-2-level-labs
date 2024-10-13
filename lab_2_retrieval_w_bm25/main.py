"""
Lab 2.

Text retrieval with BM25
"""
# pylint:disable=too-many-arguments, unused-argument
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
    tokenized_list = []
    word = []

    for element in text.lower():
        if element.isalpha():
            word.append(element)
        else:
            if word:
                tokenized_list.append(''.join(word))
                word = []
            continue

    if word:
        tokenized_list.append(''.join(word))

    return tokenized_list


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
    if not (
            isinstance(tokens, list) and
            all(isinstance(token, str) for token in tokens)
    ):
        return None
    if not (
            isinstance(stopwords, list) and
            all(isinstance(word, str) for word in stopwords)
    ):
        return None
    if not (len(tokens) and len(stopwords)):
        return None

    tokens_without_stopwords = []
    for token in tokens:
        if token not in stopwords:
            tokens_without_stopwords.append(token)

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
    if not isinstance(documents, list) or not documents:
        return None

    vocabulary = set()

    for element in documents:
        if not isinstance(element, list):
            return None
        for sub_element in element:
            if not isinstance(sub_element, str):
                return None
            vocabulary.add(sub_element)

    return list(vocabulary)


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
    if not (isinstance(vocab, list) and isinstance(document_tokens, list)):
        return None
    if not (len(vocab) and len(document_tokens)):
        return None
    if not all(isinstance(element, str) for element in vocab) or not all(
            isinstance(element, str) for element in document_tokens):
        return None

    tf_dict = {}

    for word in document_tokens:
        if word not in tf_dict:
            tf_dict[word] = 0.0
        for elem in vocab:
            if elem not in tf_dict:
                tf_dict[elem] = 0.0
            tf_dict[word] = document_tokens.count(word) / len(document_tokens)

    if not tf_dict:
        return None

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
    if not (isinstance(vocab, list) and isinstance(documents, list)):
        return None
    if not (len(vocab) and len(documents)):
        return None
    if not all(isinstance(element, str) for element in vocab):
        return None
    if not all(isinstance(element, list) and
               all(isinstance(sub_elem, str) for sub_elem in element)
               for element in documents):
        return None

    idf_dict = {}

    for doc in documents:
        for token in set(doc):
            if token in vocab:
                if not (token in idf_dict.keys()):
                    idf_dict[token] = 1
                else:
                    idf_dict[token] += 1

    idf_final = {}
    for i in idf_dict.keys():
        idf_final[i] = log((len(documents) - idf_dict[i] + 0.5) / (idf_dict[i] + 0.5))

    return idf_final


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
    if not (isinstance(tf, dict) and isinstance(idf, dict)):
        return None
    if not tf or not idf:
        return None
    if (not all(isinstance(key, str) for key in tf.keys()) or
            not all(isinstance(key, str) for key in idf.keys())):
        return None
    if (not all(isinstance(value, float) for value in tf.values()) or
            not all(isinstance(value, float) for value in idf.values())):
        return None

    tf_idf = {}
    for elem, term_frequency in tf.items():
        if elem in idf:
            tf_idf[elem] = term_frequency * idf[elem]

    if not tf_idf:
        return None

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
    if not (isinstance(vocab, list) and all(isinstance(elem, str) for elem in vocab) and vocab):
        return None
    if (not (isinstance(document, list) and all(isinstance(elem, str) for elem in document)
             and document)):
        return None

    if (not isinstance(idf_document, dict) or
            not all(isinstance(key, str) and isinstance(value, float)
                    for key, value in idf_document.items()) or not idf_document):
        return None

    if not (isinstance(k1, float) and isinstance(b, float)):
        return None

    if not (isinstance(avg_doc_len, float)) or avg_doc_len is None:
        return None

    if not (isinstance(doc_len, int)) or doc_len is None or isinstance(doc_len, bool):
        return None

    bm25_dict = {}

    for word in vocab:
        bm25_dict[word] = 0.0

    for doc_word in document:
        bm25_dict[doc_word] = 0.0

    for elem in idf_document:
        frequency = document.count(elem)
        bm25_score = (idf_document[elem] *
                      ((frequency * (k1 + 1)) / (frequency + k1 * (
                              1 - b + b * (doc_len / avg_doc_len)
                      ))))
        bm25_dict[elem] = bm25_score

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
    if (
        not isinstance(indexes, list) or
        not all(
            isinstance(elem, dict) and
            all(
                isinstance(key, str) and
                isinstance(value, float) for key, value in elem.items()
            ) for elem in indexes
        )
    ):
        return None
    if not isinstance(query, str):
        return None
    if not isinstance(stopwords, list) or not all(isinstance(elem, str) for elem in stopwords):
        return None

    tokenized_query = tokenize(query)
    if not isinstance(tokenized_query, list):
        return None

    final_query = remove_stopwords(tokenized_query, stopwords)
    if not isinstance(final_query, list):
        return None

    ranged_documents = []

    for index, index_value in enumerate(indexes):
        score = 0.0
        for word in final_query:
            if word in index_value:
                score += index_value[word]
        ranged_documents.append((index, score))

    for i in range(len(ranged_documents) - 1):
        for j in range(len(ranged_documents) - i - 1):
            if ranged_documents[j][1] < ranged_documents[j + 1][1]:
                (ranged_documents[j], ranged_documents[j + 1]) = (ranged_documents[j + 1],
                                                                  ranged_documents[j])

    if not ranged_documents:
        return None

    return ranged_documents


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
