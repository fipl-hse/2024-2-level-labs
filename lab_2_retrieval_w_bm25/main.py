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

    for letter in text:
        if not letter.isalpha() and letter != ' ':
            text = text.replace(letter, ' ')
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
    clean_tokens = []

    if any((not isinstance(tokens, list), not isinstance(stopwords, list),
            not stopwords, not tokens)):
        return None

    for word in stopwords:
        if not isinstance(word, str):
            return None

    for token in tokens:
        if not isinstance(token, str):
            clean_tokens = None
            break
        if token in stopwords:
            continue
        clean_tokens.append(token)

    return clean_tokens


def build_vocabulary(documents: list[list[str]]) -> list[str] | None:
    """
    Build a vocabulary from the documents.

    Args:
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        list[str] | None: List with unique words from the documents.

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(documents, list) or \
            not all(isinstance(document, list) for document in documents) or \
            not documents:
        return None
    for document in documents:
        if not all(isinstance(item, str) for item in document):
            return None

    result = set()
    for doc in documents:
        result |= set(doc)
    return list(result)

    if any((not isinstance(documents, list), not documents)):
        return None

    unique_words = []

    for doc in documents:
        if not isinstance(doc, list):
            return None
        for token in doc:
            if not isinstance(token, str):
                return None
            if token not in unique_words:
                unique_words.append(token)

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
    if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab) or \
            not vocab:
        return None
    if not isinstance(document_tokens, list) or \
            not all(isinstance(token, str) for token in document_tokens) or not document_tokens:
        return None

    result = {}
    for word in set(vocab) | set(document_tokens):
        result[word] = document_tokens.count(word) / len(document_tokens)
    return result

    if any((not isinstance(vocab, list), not isinstance(document_tokens, list),
            not vocab, not document_tokens)):
        return None

    frequency = {}

    for voc in vocab:
        if not isinstance(voc, str):
            return None
        frequency[voc] = 0.0
        if voc in document_tokens:
            frequency[voc] = document_tokens.count(voc) / len(document_tokens)

    for token in document_tokens:
        if not isinstance(token, str):
            return None
        if token not in frequency:
            frequency[token] = document_tokens.count(token) / len(document_tokens)

    return frequency


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
    # lab_2_retrieval_w_bm25/main.py:150:0: R0911: Too many return statements
    # (7/6) (too-many-return-statements)

    if any((not isinstance(vocab, list), not isinstance(documents, list),
            not documents, not vocab)):
        return None

    for doc in documents:
        if not isinstance(doc, list):
            return None
        for token in doc:
            if not isinstance(token, str):
                return None
    map_idf_scores = {}
    for voc in vocab:
        if not isinstance(voc, str):
            return None
        amount = 0
        for doc in documents:
            if voc in doc:
                amount += 1
        idf = log((len(documents) - amount + 0.5) / (amount + 0.5))
        map_idf_scores[voc] = idf

    return map_idf_scores


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
    if not tf or not isinstance(tf, dict) or not all(isinstance(key, str) for key in tf) or \
            not all(isinstance(value, float) for value in tf.values()):
        return None
    if not idf or not isinstance(idf, dict) or not all(isinstance(key, str) for key in idf) or \
            not all(isinstance(value, float) for value in idf.values()):
        return None

    tf_idf = {}
    for word in tf:
        tf_idf[word] = tf[word] * idf[word]
    return tf_idf

    if any((not isinstance(tf, dict), not isinstance(idf, dict), not idf, not tf)):
        return None
    for t_it in tf:
        if any((not isinstance(t_it, str), not isinstance(tf[t_it], float))):
            return None
    for i_it in idf:
        if any((not isinstance(i_it, str), not isinstance(idf[i_it], float))):
            return None

    map_scores = {}

    for t in tf.keys():
        score = tf[t] * idf[t]
        map_scores[t] = score

    return map_scores


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
    if not vocab or not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
        return None
    if not document or not isinstance(document, list) \
            or not all(isinstance(item, str) for item in document):
        return None
    if not idf_document or not isinstance(idf_document, dict) \
            or not all(isinstance(key, str) for key in idf_document) \
            or not all(isinstance(value, float) for value in idf_document.values()):
        return None
    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) \
            or isinstance(doc_len, bool) or not isinstance(k1, float) or not isinstance(b, float):
        return None

    bm25 = {}
    for word in set(vocab) | set(document):
        if word in idf_document:
            word_count = document.count(word)
            bm25[word] = idf_document[word] * ((word_count * (k1 + 1)) / (
                    word_count + k1 * (1 - b + (b * doc_len / avg_doc_len))))
        else:
            bm25[word] = 0.0
    return bm25

    if any((not vocab, not isinstance(vocab, list),
            not document, not isinstance(document, list),
            not idf_document, not isinstance(idf_document, dict))):
        return None

    for item in vocab:
        if not isinstance(item, str):
            return None

    for item in document:
        if not isinstance(item, str):
            return None

    for item in idf_document:
        if not isinstance(item, str) or not isinstance(idf_document[item], float):
            return None

    if not isinstance(avg_doc_len, float) or not isinstance(doc_len, int) \
            or isinstance(doc_len, bool) or not isinstance(k1, float) \
            or not isinstance(b, float):
        return None

    bm5_doc = {}

    for voc in vocab:
        freq = document.count(voc)
        bm5_doc[voc] = (idf_document[voc] * freq * (k1 + 1)
                        / (freq + k1 * (1 - b + b * (doc_len / avg_doc_len))))

    for tok in document:
        bm5_doc[tok] = 0.0
        if tok in vocab:
            freq = document.count(tok)
            bm5_doc[tok] = (idf_document[tok] * freq * (k1 + 1)
                            / (freq + k1 * (1 - b + b * ((abs(doc_len)) / avg_doc_len))))

    return bm5_doc


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
    if any((not isinstance(indexes, list), not isinstance(query, str),
            not isinstance(stopwords, list), query == 'no query',
            not query, not indexes, not stopwords)):
        return None
    if any(not isinstance(index, dict) for index in indexes) or \
        any(any(not isinstance(ind, str) or not isinstance(index[ind], float)
                for ind in index)for index in indexes) or \
        any(not isinstance(stopword, str) for stopword in stopwords):
        return None

    tokenized_query = tokenize(query)
    if tokenized_query is None:
        return None
    clean_query = remove_stopwords(tokenized_query, stopwords)
    if clean_query is None:
        return None

    uniq_query = []
    for qu in clean_query:
        if qu not in uniq_query:
            uniq_query.append(qu)

    # unsorted_ranking_map = {}
    #
    # for i, doc in enumerate(indexes):
    #     ranking_word = 0.0
    #     for word in uniq_query:
    #         if word in doc:
    #             ranking_word += doc[word]
    #     unsorted_ranking_map[i] = ranking_word

    unsorted_ranking_map = {
        i: sum(doc.get(word, 0) for word in uniq_query)
        for i, doc in enumerate(indexes)
    }

    sorted_dict = dict(sorted(unsorted_ranking_map.items(), key=lambda item: item[1], reverse=True))

    return [(sort_doc, sorted_dict[sort_doc]) for sort_doc in sorted_dict]



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
    if not vocab or not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab)\
            or not document or not isinstance(document, list):
        return None
    if not all(isinstance(item, str) for item in document) or not idf_document \
            or not isinstance(idf_document, dict) \
            or not all(isinstance(key, str) for key in idf_document) \
            or not all(isinstance(value, float) for value in idf_document.values()):
        return None
    if not isinstance(alpha, float) or not isinstance(k1, float) \
            or not isinstance(b, float) or not isinstance(avg_doc_len, float):
        return None
    if not isinstance(doc_len, int) or isinstance(doc_len, bool) or doc_len < 0:
        return None

    bm25_with_cutoff = {}
    for word in vocab:
        if word in idf_document and idf_document[word] >= alpha:
            word_count = document.count(word)
            bm25_with_cutoff[word] = idf_document[word] * ((word_count * (k1 + 1)) / (
                    word_count + k1 * (1 - b + (b * doc_len / avg_doc_len))))
    return bm25_with_cutoff


def save_index(index: list[dict[str, float]], file_path: str) -> None:
    """
    Save the index to a file.

    Args:
        index (list[dict[str, float]]): The index to save.
        file_path (str): The path to the file where the index will be saved.
    """
    if not index or not isinstance(index, list) or \
            not all(isinstance(item, dict) for item in index) or \
            not all(isinstance(key, str) for item in index for key in item) or \
            not all(isinstance(value, float) for item in index for value in item.values()):
        return None
    if not isinstance(file_path, str) or not file_path:
        return None

    with open(file_path, 'w', encoding='utf-8') as file:
        dump(index, file)
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
    if not file_path or not isinstance(file_path, str):
        return None

    with open(file_path, 'r', encoding='utf-8') as file:
        index: list[dict[str, float]] = load(file)
    return index


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
    if not rank or not isinstance(rank, list) or not all(isinstance(item, int) for item in rank):
        return None
    if not golden_rank or not isinstance(golden_rank, list) or \
            not all(isinstance(item, int) for item in golden_rank) or \
            len(rank) != len(golden_rank):
        return None

    n = len(rank)
    rank_differences = 0
    for item in rank:
        if item in golden_rank:
            rank_differences += (golden_rank.index(item) - rank.index(item)) ** 2
    return 1 - (6 * rank_differences) / (n * (n**2 - 1))
