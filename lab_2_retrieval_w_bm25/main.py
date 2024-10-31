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
    for letter in text:
        if not letter.isalpha():
            text = text.replace(letter,' ')
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
    if not tokens or not isinstance(tokens, list) or not\
        all(isinstance(token, str) for token in tokens):
        return None
    if not stopwords or not isinstance(stopwords, list) or not\
        all(isinstance(stopword, str) for stopword in stopwords):
        return None
    return [var for var in tokens if var not in stopwords]


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents,list) or not\
            all(isinstance(document,list) for document in documents):
        return None
    for document in documents:
        if not all(isinstance(word,str) for word in document):
            return None
    vocabulary = {word for document in documents for word in document
                  if isinstance(word,str)}
    if not vocabulary:
        return None
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
    if (not vocab or not document_tokens or not isinstance(vocab, list) or
            not isinstance(document_tokens, list)):
        return None
    if (not all(isinstance(term, str) for term in vocab) or
            not all(isinstance(token, str) for token in document_tokens)):
        return None
    tf_dict = {word: (document_tokens.count(word) / len(document_tokens)) for
               word in set(vocab + document_tokens)}
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
    if not documents or not vocab or not isinstance(documents, list) or not isinstance(vocab, list):
        return None
    if (not all(isinstance(document, list) for document in documents) or
            not all(isinstance(word, str) for word in vocab)):
        return None
    for document in documents:
        if not all(isinstance(token, str) for token in document):
            return None
    idf_dict = {}
    for word in vocab:
        check_count = 0
        for document in documents:
            if word in document:
                check_count += 1
        idf_dict[word] = math.log((len(documents) - check_count + 0.5) / (check_count + 0.5))
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
    if not tf or not idf or not isinstance(tf, dict) or not isinstance(idf, dict):
        return None
    if not all(isinstance(word, str) for word in idf):
        return None
    tf_idf = {word: tf[word] * idf[word] for word in tf if word in idf}
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
    if (not isinstance(vocab, list) or not all(isinstance(word, str) for word in vocab)
            or not isinstance(document, list) or not
            all(isinstance(word, str) for word in document) or not isinstance(k1, float)):
        return None
    if (not isinstance(b, float) or not isinstance(avg_doc_len, float)
    or not isinstance(doc_len, int) or not isinstance(idf_document, dict)
    or isinstance(doc_len, bool)):
        return None
    if not all((isinstance(key, str) and isinstance(value, float) for key, value in
                idf_document.items())):
        return None
    if not vocab or not document or not idf_document:
        return None

    bm25 = {}
    for word in set(vocab + document):
        if word not in idf_document:
            bm25[word] = 0.0
        else:
            bm25[word] = (idf_document[word] * document.count(word) * (k1 + 1) /
                              (document.count(word) + k1 * (1 - b + b * doc_len / avg_doc_len)))
    if not bm25:
        return None
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
    if not isinstance(indexes,list) or not isinstance(query, str) or\
        not isinstance(stopwords,list):
        return None
    if not all(isinstance(index,dict) for index in indexes) or\
        not all(isinstance(stopword,str) for stopword in stopwords):
        return None
    tokenized_query = tokenize(query)
    if not tokenized_query:
        return None
    clean_query = remove_stopwords(tokenized_query, stopwords)
    if not clean_query:
        return None
    results = []
    for num,document in enumerate(indexes):
        token_values_sum = sum(document[token] for token in clean_query if token
                               in document)
        results.append((num,token_values_sum))
    results.sort(key=lambda x: x[-1], reverse=True)
    if not results:
        return None
    return results



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
    if (not isinstance(vocab, list) or not all(isinstance(word, str) for word in vocab)
            or not isinstance(document, list) or not
            all(isinstance(word, str) for word in document) or not isinstance(k1, float)):
        return None
    if (not isinstance(b, float) or not isinstance(avg_doc_len, float)
    or not isinstance(doc_len, int) or not isinstance(idf_document, dict)
    or isinstance(doc_len, bool)):
        return None
    if not all((isinstance(key, str) and isinstance(value, float) for key, value in
                idf_document.items())):
        return None
    if not isinstance(alpha,float) or doc_len < 0 or not vocab\
            or not document or not idf_document:
        return None
    bm25_wcoff = {}
    for word in vocab:
        if idf_document[word] >= alpha:
            bm25_wcoff[word] = (idf_document[word] * document.count(word) * (k1 + 1) /
                          (document.count(word) + k1 * (1 - b + b * doc_len / avg_doc_len)))
    return bm25_wcoff


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
