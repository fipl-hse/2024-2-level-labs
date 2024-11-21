"""
Lab 3.

Vector search with text retrieving
"""
import json
from math import sqrt
# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from typing import Protocol

from lab_2_retrieval_w_bm25.main import calculate_idf, calculate_tf

Vector = tuple[float, ...]
"Type alias for vector representation of a text."


class NodeLike(Protocol):
    """
    Type alias for a tree node.
    """

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: State of the Node instance
        """

    def load(self, state: dict) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict): Saved state of the Node

        Returns:
            bool: True if Node was loaded successfully, False in other cases
        """


def calculate_distance(query_vector: Vector, document_vector: Vector) -> float | None:
    """
    Calculate Euclidean distance for a document vector.

    Args:
        query_vector (Vector): Vectorized query
        document_vector (Vector): Vectorized documents

    Returns:
        float | None: Euclidean distance for vector

    In case of corrupt input arguments, None is returned.
    """
    if not (isinstance(query_vector, (list, tuple)) and isinstance(document_vector, (list, tuple))
            and all(isinstance(cor, float) for cor in query_vector) and
            all(isinstance(cor, float) for cor in document_vector) and
            (len(query_vector) == len(document_vector) or len(query_vector) == 0 or len(
                document_vector) == 0)):
        return None
    if not query_vector or not document_vector:
        return 0.0
    distance: float = 0.0
    for idx, qcor in enumerate(query_vector):
        distance += (qcor - document_vector[idx]) ** 2
    return sqrt(distance)


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """
    state: dict = {
        "len": len(vector),
        "elements": {}
    }
    for idx, el in enumerate(vector):
        if el != 0:
            state["elements"][idx] = el
    return state


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if not (isinstance(state, dict) and "len" in state and "elements" in state and isinstance(
            state["len"], int) and isinstance(state["elements"], dict)):
        return None
    vect = list(0.0 for _ in range(state["len"]))
    for idx in state["elements"]:
        vect[int(idx)] = state["elements"][idx]
    return Vector(vect)


class Tokenizer:
    """
    Tokenizer with removing stop words.
    """

    _stop_words: list[str]

    def __init__(self, stop_words: list[str]) -> None:
        """
        Initialize an instance of the Tokenizer class.

        Args:
            stop_words (list[str]): List with stop words
        """
        self._stop_words = stop_words

    def tokenize(self, text: str) -> list[str] | None:
        """
        Tokenize the input text into lowercase words without punctuation, digits and other symbols.

        Args:
            text (str): The input text to tokenize

        Returns:
            list[str] | None: A list of words from the text

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(text, str):
            return None
        out = []
        word = ''
        for char in text.lower():
            if char.isalpha():
                word += char
            else:
                if word != '':
                    out.append(word)
                word = ''
        return self._remove_stop_words(out) or None

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if (not documents or not isinstance(documents, list) or
                not all(isinstance(itm, str) for itm in documents)):
            return None
        out = []
        for item in documents:
            if not item:
                return None
            if not self.tokenize(item):
                return None
            out.append(self.tokenize(item))
        return out

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if (not tokens or not isinstance(tokens, list) or
                not all(isinstance(token, str) for token in tokens)):
            return None
        if (not self._stop_words or not
                isinstance(self._stop_words, list) or
                not all(isinstance(word, str) for word
                in self._stop_words)):
            return None
        srtd = []
        for token in tokens:
            if token not in self._stop_words:
                srtd.append(token)
        return srtd


class Vectorizer:
    """
    TF-IDF Vectorizer.
    """

    _corpus: list[list[str]]
    _idf_values: dict[str, float]
    _vocabulary: list[str]
    _token2ind: dict[str, int]

    def __init__(self, corpus: list[list[str]]) -> None:
        """
        Initialize an instance of the Vectorizer class.

        Args:
            corpus (list[list[str]]): Tokenized documents to vectorize
        """
        self._corpus = corpus
        self._idf_values = {}
        self._vocabulary = []
        self._token2ind = {}

    def build(self) -> bool:
        """
        Build vocabulary with tokenized_documents.

        Returns:
            bool: True if built successfully, False in other case
        """
        if not (isinstance(self._corpus, list) and self._corpus):
            return False
        for tknzd in self._corpus:
            for tkn in tknzd:
                if tkn not in self._vocabulary:
                    self._vocabulary.append(tkn)
        self._vocabulary.sort()
        for tkn in self._vocabulary:
            self._token2ind[tkn] = self._vocabulary.index(tkn)
        idf = calculate_idf(self._vocabulary, self._corpus)
        if idf is None or self._vocabulary is None:
            return False
        self._idf_values = idf
        return True

    def vectorize(self, tokenized_document: list[str]) -> Vector | None:
        """
        Create a vector for tokenized document.

        Args:
            tokenized_document (list[str]): Tokenized document to vectorize

        Returns:
            Vector | None: TF-IDF vector for document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(tokenized_document, list) and tokenized_document and all(
                isinstance(tkn, str) for tkn in tokenized_document)):
            return None
        return self._calculate_tf_idf(tokenized_document)

    def vector2tokens(self, vector: Vector) -> list[str] | None:
        """
        Recreate a tokenized document based on a vector.

        Args:
            vector (Vector): Vector to decode

        Returns:
            list[str] | None: Tokenized document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(vector, tuple) and all(isinstance(cor, float) for cor in vector) and len(
                vector) == len(self._vocabulary)):
            return None
        doc = []
        for tkn in self._vocabulary:
            if vector[self._token2ind[tkn]]:
                doc.append(tkn)
        return doc

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """
        if not isinstance(file_path, str):
            return False
        stats = {
            "idf_values": self._idf_values,
            "vocabulary": self._vocabulary,
            "token2ind": self._token2ind
        }
        with open(file_path, "w", encoding="UTF-8") as file:
            json.dump(stats, file, ensure_ascii=False, indent=4)
        return True

    def load(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if the vectorizer was saved successfully

        In case of corrupt input arguments, False is returned.
        """
        if not isinstance(file_path, str):
            return False
        with open(file_path, "r", encoding="UTF-8") as file:
            stats = json.load(file)
        if not ("vocabulary" in stats and "token2ind" in stats
                and "idf_values" in stats):
            return False
        self._vocabulary = stats["vocabulary"]
        self._token2ind = stats["token2ind"]
        self._idf_values = stats["idf_values"]
        return True

    def _calculate_tf_idf(self, document: list[str]) -> Vector | None:
        """
        Get TF-IDF for document.

        Args:
            document (list[str]): Tokenized document to vectorize

        Returns:
            Vector | None: TF-IDF vector for document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(document, list) and document and all(
                isinstance(tkn, str) for tkn in document)):
            return None
        vect = list(0.0 for _ in range(len(self._vocabulary)))
        for token in document:
            if self._token2ind.get(token) is not None:
                tf: dict[str, float] | None = calculate_tf(self._vocabulary, document)
                if tf is not None:
                    vect[self._token2ind[token]] = tf[token] * self._idf_values[token]
        return Vector(vect)


class BasicSearchEngine:
    """
    Engine based on KNN algorithm.
    """

    _vectorizer: Vectorizer
    _tokenizer: Tokenizer
    _documents: list[str]
    _document_vectors: list[Vector]

    def __init__(self, vectorizer: Vectorizer, tokenizer: Tokenizer) -> None:
        """
        Initialize an instance of the BasicSearchEngine class.

        Args:
            vectorizer (Vectorizer): Vectorizer for documents vectorization
            tokenizer (Tokenizer): Tokenizer for tokenization
        """
        self._vectorizer = vectorizer
        self._tokenizer = tokenizer
        self._documents = []
        self._document_vectors = []

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for engine.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if documents are successfully indexed

        In case of corrupt input arguments, False is returned.
        """
        if not (isinstance(documents, list) and documents and all(
                isinstance(doc, str) for doc in documents)):
            return False
        self._documents = documents
        for doc in documents:
            iddoc: tuple[float, ...] | None = self._index_document(doc)
            if iddoc is not None:
                self._document_vectors.append(iddoc)
            else:
                return False
        return None not in self._document_vectors

    def retrieve_relevant_documents(
        self, query: str, n_neighbours: int
    ) -> list[tuple[float, str]] | None:
        """
        Index documents for retriever.

        Args:
            query (str): Query for obtaining relevant documents
            n_neighbours (int): Number of relevant documents to return

        Returns:
            list[tuple[float, str]] | None: Relevant documents with their distances

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(query, str) and isinstance(n_neighbours, int)):
            return None
        tqr = self._tokenizer.tokenize(query)
        if tqr is None:
            return None
        vqr = self._vectorizer.vectorize(tqr)
        if vqr is None:
            return None
        rng_am = self._calculate_knn(vqr, self._document_vectors, n_neighbours)
        if not rng_am:
            return None
        out = []
        for rng in rng_am:
            out.append((rng[1], self._documents[rng[0]]))
        return out

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where to save the instance

        Returns:
            bool: returns True if save was done correctly, False in another cases
        """
        if not isinstance(file_path, str):
            return False
        state = {
            "engine": self._dump_documents()
        }
        with open(file_path, "w", encoding="UTF-8") as file:
            json.dump(state, file, indent=4, ensure_ascii=False)
        return True

    def load(self, file_path: str) -> bool:
        """
        Load engine from state.

        Args:
            file_path (str): The path to the file with state

        Returns:
            bool: True if engine was loaded, False in other cases
        """
        if not isinstance(file_path, str):
            return False
        with open(file_path, "r", encoding="UTF-8") as file:
            state = json.load(file)
        if "engine" not in state:
            return False
        return self._load_documents(state["engine"])

    def retrieve_vectorized(self, query_vector: Vector) -> str | None:
        """
        Retrieve document by vector.

        Args:
            query_vector (Vector): Question vector

        Returns:
            str | None: Answer document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(query_vector, tuple) and all(
                isinstance(cor, float) for cor in query_vector)):
            return None
        knn: list[tuple[int, float]] | None = self._calculate_knn(query_vector,
                                                                  self._document_vectors, 1)
        if knn is None or not knn:
            return None
        return self._documents[knn[0][0]]

    def _calculate_knn(
        self, query_vector: Vector, document_vectors: list[Vector], n_neighbours: int
    ) -> list[tuple[int, float]] | None:
        """
        Find nearest neighbours for a query vector.

        Args:
            query_vector (Vector): Vectorized query
            document_vectors (list[Vector]): Vectorized documents
            n_neighbours (int): Number of neighbours to return

        Returns:
            list[tuple[int, float]] | None: Nearest neighbours indices and distances

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(query_vector, (tuple, list))
                and all(isinstance(cor, float) for cor in query_vector)
                and isinstance(document_vectors, list)
                and all(isinstance(vect, (tuple, list))
                        and all(isinstance(cor, float) for cor in vect) for vect in
                        document_vectors)
                and isinstance(n_neighbours, int) and document_vectors):
            return None
        rngs: list[tuple[int, float]] = []
        for document_vector in document_vectors:
            rng: float | None = calculate_distance(query_vector, document_vector)
            if rng is None:
                return None
            rngs.append((document_vectors.index(document_vector), rng))
        rngs.sort(key=lambda dist: dist[1])
        return rngs[:n_neighbours]

    def _index_document(self, document: str) -> Vector | None:
        """
        Index document.

        Args:
            document (str): Document to index

        Returns:
            Vector | None: Returns document vector

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(document, str):
            return None
        doc: list[str] | None = self._tokenizer.tokenize(document)
        if doc is None:
            return None
        return self._vectorizer.vectorize(doc)

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        return {
            "documents": self._documents,
            "document_vectors": [save_vector(vector) for vector in self._document_vectors]
        }

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if not (isinstance(state, dict) and "documents" in state and "document_vectors" in state):
            return False
        self._documents = state["documents"]
        load = [load_vector(vector) for vector in state["document_vectors"]]
        for vect in load:
            if vect is None:
                return False
            self._document_vectors.append(vect)
        if None in self._document_vectors:
            return False
        return True


class Node(NodeLike):
    """
    Interface definition for Node for KDTree.
    """

    vector: Vector
    payload: int
    left_node: NodeLike | None
    right_node: NodeLike | None

    def __init__(
        self,
        vector: Vector = (),
        payload: int = -1,
        left_node: NodeLike | None = None,
        right_node: NodeLike | None = None,
    ) -> None:
        """
        Initialize an instance of the Node class.

        Args:
            vector (Vector): Current vector node
            payload (int): Index of current vector
            left_node (NodeLike | None): Left node
            right_node (NodeLike | None): Right node
        """

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: state of the Node instance
        """

    def load(self, state: dict[str, dict | int]) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict[str, dict | int]): Saved state of the Node

        Returns:
            bool: True if Node was loaded successfully, False in other cases.
        """


class NaiveKDTree:
    """
    NaiveKDTree.
    """

    _root: NodeLike | None

    def __init__(self) -> None:
        """
        Initialize an instance of the KDTree class.
        """

    def build(self, vectors: list[Vector]) -> bool:
        """
        Build tree.

        Args:
            vectors (list[Vector]): Vectors for tree building

        Returns:
            bool: True if tree was built, False in other cases

        In case of corrupt input arguments, False is returned.
        """

    def query(self, vector: Vector, k: int = 1) -> list[tuple[float, int]] | None:
        """
        Get k nearest neighbours for vector.

        Args:
            vector (Vector): Vector to get k nearest neighbours
            k (int): Number of nearest neighbours to get

        Returns:
            list[tuple[float, int]] | None: Nearest neighbours indices

        In case of corrupt input arguments, None is returned.
        """

    def save(self) -> dict | None:
        """
        Save NaiveKDTree instance to state.

        Returns:
            dict | None: state of the NaiveKDTree instance

        In case of corrupt input arguments, None is returned.
        """

    def load(self, state: dict) -> bool:
        """
        Load NaiveKDTree instance from state.

        Args:
            state (dict): saved state of the NaiveKDTree

        Returns:
            bool: True is loaded successfully, False in other cases
        """

    def _find_closest(self, vector: Vector, k: int = 1) -> list[tuple[float, int]] | None:
        """
        Get k nearest neighbours for vector by filling best list.

        Args:
            vector (Vector): Vector for getting knn
            k (int): The number of nearest neighbours to return

        Returns:
            list[tuple[float, int]] | None: The list of k nearest neighbours

        In case of corrupt input arguments, None is returned.
        """


class KDTree(NaiveKDTree):
    """
    KDTree.
    """

    def _find_closest(self, vector: Vector, k: int = 1) -> list[tuple[float, int]] | None:
        """
        Get k nearest neighbours for vector by filling best list.

        Args:
            vector (Vector): Vector for getting knn
            k (int): The number of nearest neighbours to return

        Returns:
            list[tuple[float, int]] | None: The list of k nearest neighbours

        In case of corrupt input arguments, None is returned.
        """


class SearchEngine(BasicSearchEngine):
    """
    Retriever based on KDTree algorithm.
    """

    _tree: NaiveKDTree

    def __init__(self, vectorizer: Vectorizer, tokenizer: Tokenizer) -> None:
        """
        Initialize an instance of the SearchEngine class.

        Args:
            vectorizer (Vectorizer): Vectorizer for documents vectorization
            tokenizer (Tokenizer): Tokenizer for tokenization
        """

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for retriever.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if document is successfully indexed

        In case of corrupt input arguments, False is returned.
        """

    def retrieve_relevant_documents(
        self, query: str, n_neighbours: int = 1
    ) -> list[tuple[float, str]] | None:
        """
        Index documents for retriever.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]] | None: Relevant documents with their distances.

        In case of corrupt input arguments, None is returned.
        """

    def save(self, file_path: str) -> bool:
        """
        Save the SearchEngine instance to a file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """

    def load(self, file_path: str) -> bool:
        """
        Load a SearchEngine instance from a file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if engine was loaded successfully, False in other cases
        """


class AdvancedSearchEngine(SearchEngine):
    """
    Retriever based on KDTree algorithm.
    """

    _tree: KDTree

    def __init__(self, vectorizer: Vectorizer, tokenizer: Tokenizer) -> None:
        """
        Initialize an instance of the AdvancedSearchEngine class.

        Args:
            vectorizer (Vectorizer): Vectorizer for documents vectorization
            tokenizer (Tokenizer): Tokenizer for tokenization
        """
