"""
Lab 2.

Text retrieval with BM25
"""
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
    if not isinstance(text, str):
        return None

    text = text.lower()

    for symb in text:
        if not symb.isalpha() and symb != ' ':
            text = text.replace(symb, ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')


    text = text.strip()

    tokens = text.split(' ')

    if tokens == ['']:
        tokens = []

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
    clean_tokens = []

    if not isinstance(tokens, list):
        return None

    if any((not isinstance(stopwords, list), not stopwords)):
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
    if not clean_tokens:
        return None
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

    if not isinstance(documents, list):
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
    if unique_words:
        return unique_words
    return None

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

    if any((not isinstance(vocab, list), not isinstance(document_tokens, list))):
        return None

    if len(vocab) == 0 or len(document_tokens) == 0:
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
    if frequency:
        return frequency
    return None

def calculate_idf(vocab: list[str], documents: list[list[str]]) -> dict[str, float] | None:
    from math import log
    """
    Calculate inverse document frequency for each term in the vocabulary.

    Args:
        vocab (list[str]): Vocabulary list.
        documents (list[list[str]]): List of tokenized documents.

    Returns:
        dict[str, float] | None: Mapping from vocabulary terms to its IDF scores.

    In case of corrupt input arguments, None is returned.
    """
    if any((not isinstance(vocab, list), not isinstance(documents, list))):
        return None

    for doc in documents:
        if not isinstance(doc, list):
            return None
        for token in doc:
            if not isinstance(token, str):
                return None
    len_doc = len(documents)
    if len_doc == 0:
        return None
    map_idf_scores = {}

    for voc in vocab:
        if not isinstance(voc, str):
            return None

        amount = 0
        for doc in documents:
            if voc in doc:
                amount += 1

        idf = log((len_doc - amount + 0.5) / (amount + 0.5))
        map_idf_scores[voc] = idf
    if map_idf_scores:
        return map_idf_scores
    return None


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
    if any((not isinstance(tf, dict), not isinstance(idf, dict))):
        return None

    for t_k in tf.keys():
        if not isinstance(t_k, str):
            return None

    for i_k in idf.keys():
        if not isinstance(i_k, str):
            return None

    for t_v in tf.values():
        if not isinstance(t_v, float):
            return None

    for i_v in idf.values():
        if not isinstance(i_v, float):
            return None

    if not idf or not tf:
        return None

    map_scores = {}

    for t in tf.keys():
        score = tf[t] * idf[t]
        map_scores[t] = score
    if map_scores:
        return map_scores

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

    if any((not isinstance(vocab, list), not isinstance(document, list))):
        return None

    if any((not isinstance(idf_document, dict), not isinstance(k1, float),
            not isinstance(b, float))):
        return None

    if any((not isinstance(avg_doc_len, float), not isinstance(doc_len, int), isinstance(doc_len, bool))):
        return None

    if any((vocab == [], idf_document == {}, document == [])):
        return None

    for voc in vocab:
        if not isinstance(voc, str):
            return None
        if voc.split(' ')[0] != voc:
            return None

    for token in document:
        if not isinstance(token, str):
            return None
        if token.split(' ')[0] != token:
            return None

    for pair in idf_document:
        if not (isinstance(pair, str) and isinstance(idf_document[pair], float)):
            return None
        if pair.split() != [pair]:
            return None

    bm5_doc = {}

    for voc in vocab:
        freq = document.count(voc)
        bm5_doc[voc] = (idf_document[voc] * freq * (k1 + 1)
                    / (freq + k1 * (1 - b + b * ((abs(doc_len)) / avg_doc_len))))

    for tok in document:
        bm5_doc[tok] = 0.0
        if tok in vocab:
            freq = document.count(tok)
            bm5_doc[tok] = (idf_document[tok] * freq * (k1 + 1)
                    / (freq + k1 * (1 - b + b * ((abs(doc_len)) / avg_doc_len))))

    # step last: cleaning
    if not bm5_doc:
        return None
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
            not isinstance(stopwords, list))):
        return None

    if not query or query == 'no query':
        return None

    for stopword in stopwords:
        if not isinstance(stopword, str):
            return None

    for index in indexes:
        if not isinstance(index, dict):
            return None
        for ind in index:
            if any((not isinstance(ind, str), not isinstance(index[ind], float))):
                return None

    tokenized_query = remove_stopwords(tokenize(query), stopwords)

    uniq_query = []
    for qu in tokenized_query:
        if qu not in uniq_query:
            uniq_query.append(qu)

    unsorted_ranking_map = {}

    for i, doc in enumerate(indexes):
        ranking_word = 0
        for word in uniq_query:
            if word in doc.keys():
                ranking_word += doc[word]
        unsorted_ranking_map[i] = ranking_word

    sorted_dict = dict(sorted(unsorted_ranking_map.items(), key=lambda item: item[1], reverse=True))
    sorted_ranking_map = []

    for sort_doc in sorted_dict:
        sorted_ranking_map.append((sort_doc, sorted_dict[sort_doc]))

    if not sorted_ranking_map:
        return None
    return sorted_ranking_map


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
