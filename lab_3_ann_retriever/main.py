"""
Lab 3.

Vector search with text retrieving
"""

import math
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

    if query_vector is None or document_vector is None:
        return None
    if not query_vector or not document_vector:
        return 0.0
    return math.sqrt(sum((q - d) ** 2 for q, d in zip(query_vector, document_vector)))


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """


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
        for char in text:
            if not char.isalpha() and char != ' ':
                text = text.replace(char, ' ')
        return self._remove_stop_words(text.lower().split())

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """

        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
            return None
        tokenized_docs = [self.tokenize(doc) for doc in documents]
        if any(doc is None for doc in tokenized_docs):
            return None
        return [doc for doc in tokenized_docs if doc]

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """

        if (not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens)
                or not tokens):
            return None
        return [token for token in tokens if token not in self._stop_words]


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

        if not self._corpus or not isinstance(self._corpus, list):
            return False
        unique_terms = set(token for doc in self._corpus for token in doc)
        self._vocabulary = sorted(unique_terms)
        self._token2ind = {word: index for index, word in enumerate(self._vocabulary)}
        self._idf_values = calculate_idf(self._vocabulary, self._corpus) or {}
        if self._idf_values is None or not self._idf_values:
            return False
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

        if not (isinstance(tokenized_document, list) and tokenized_document and
                all(isinstance(token, str) for token in tokenized_document)):
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
        if not isinstance(vector, tuple) or len(vector) != len(self._vocabulary):
            return None
        tokens = [self._vocabulary[i] for i, value in enumerate(vector) if value > 0]
        if not tokens:
            return None
        return tokens

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """

    def load(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if the vectorizer was saved successfully

        In case of corrupt input arguments, False is returned.
        """

    def _calculate_tf_idf(self, document: list[str]) -> Vector | None:
        """
        Get TF-IDF for document.

        Args:
            document (list[str]): Tokenized document to vectorize

        Returns:
            Vector | None: TF-IDF vector for document

        In case of corrupt input arguments, None is returned.
        """

        if (not isinstance(document, list) or not document or
                not all(isinstance(token, str) for token in document)):
            return None
        if self._idf_values is None:
            return None
        tf_idf_vector = [0.0] * len(self._vocabulary)
        for token in document:
            if self._token2ind.get(token) is not None:
                tf_values = calculate_tf(self._vocabulary, document)
                if tf_values is not None:
                    tf_idf_vector[self._token2ind[token]] = (
                            tf_values[token] * self._idf_values[token])
        return Vector(tf_idf_vector)


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
        self._documents = []
        self._document_vectors = []
        for doc in documents:
            vector = self._index_document(doc)
            if vector is None:
                return False
            self._documents.append(doc)
            self._document_vectors.append(vector)
        return True

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
        tokenized_query = self._tokenizer.tokenize(query)
        if tokenized_query is None:
            return None
        query_vector = self._vectorizer.vectorize(tokenized_query)
        if query_vector is None:
            return None
        nearest_neighbours = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if not nearest_neighbours or any(neighbour is None or neighbour[0] is None or neighbour[1]
                                         is None for neighbour in nearest_neighbours):
            return None
        relevant_documents = []
        for document_index, document_distance in nearest_neighbours:
            text = self._documents[document_index]
            relevant_documents.append((document_distance, text))
        return relevant_documents

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where to save the instance

        Returns:
            bool: returns True if save was done correctly, False in another cases
        """

    def load(self, file_path: str) -> bool:
        """
        Load engine from state.

        Args:
            file_path (str): The path to the file with state

        Returns:
            bool: True if engine was loaded, False in other cases
        """

    def retrieve_vectorized(self, query_vector: Vector) -> str | None:
        """
        Retrieve document by vector.

        Args:
            query_vector (Vector): Question vector

        Returns:
            str | None: Answer document

        In case of corrupt input arguments, None is returned.
        """

        if not isinstance(query_vector, (list, tuple)) or not all(
                isinstance(val, (int, float)) for val in query_vector):
            return None
        if not self._document_vectors or any(len(query_vector) != len(doc_vector)
                                             for doc_vector in self._document_vectors):
            return None
        nearest_neighbors = self._calculate_knn(query_vector, self._document_vectors, 1)
        if nearest_neighbors is None or not nearest_neighbors:
            return None
        return self._documents[nearest_neighbors[0][0]]

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

        bad_input = (not isinstance(query_vector, (list, tuple)) or not query_vector
                     or not isinstance(document_vectors, list) or not document_vectors
                     or not isinstance(n_neighbours, int) or n_neighbours <= 0)
        if bad_input:
            return None
        distances = []
        for doc_vector in document_vectors:
            distance = calculate_distance(query_vector, doc_vector)
            if distance is not None:
                distances.append((document_vectors.index(doc_vector), distance))
        distances.sort(key=lambda dist: dist[1])
        return distances[:n_neighbours]

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
        tokenized_document = self._tokenizer.tokenize(document)
        if tokenized_document is None or not tokenized_document:
            return None
        document_vector = self._vectorizer.vectorize(tokenized_document)
        if document_vector is None:
            return None
        return document_vector

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """


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

        self.vector = vector
        self.payload = payload
        self.left_node = left_node
        self.right_node = right_node

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

        self._root = None

    def build(self, vectors: list[Vector]) -> bool:
        """
        Build tree.

        Args:
            vectors (list[Vector]): Vectors for tree building

        Returns:
            bool: True if tree was built, False in other cases

        In case of corrupt input arguments, False is returned.
        """

        if not vectors or not all(isinstance(vec, tuple) and len(vec) > 0 for vec in vectors):
            return False
        dimensions = len(vectors[0])
        another_vec = vectors[:]
        nodes = [(vectors, 0, None, True)]
        while nodes:
            current_vectors, depth, parent, is_left = nodes.pop()
            if not current_vectors:
                continue
            axis = depth % dimensions
            try:
                current_vectors.sort(key=lambda x: x[axis])
            except (IndexError, TypeError):
                return False
            median_index = len(current_vectors) // 2
            median_point = current_vectors[median_index]
            try:
                node_median = Node(median_point, another_vec.index(median_point))
            except ValueError:
                return False
            if parent is None:
                self._root = node_median
            else:
                if is_left:
                    parent.left_node = node_median
                else:
                    parent.right_node = node_median
            nodes.append((current_vectors[:median_index], depth + 1, node_median, True))
            nodes.append((current_vectors[median_index + 1:], depth + 1, node_median, False))
        return self._root is not None

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

        if not vector or not k:
            return None
        return self._find_closest(vector, k)

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

        if not (isinstance(vector, tuple) and isinstance(k, int) and vector
                and self._root is not None and isinstance(self._root, Node)):
            return None
        best = []
        nodes = [(self._root, 0)]
        while nodes:
            node, depth = nodes.pop()
            if node is None:
                continue
            dist = calculate_distance(vector, node.vector)
            if dist is None:
                continue
            best.append((dist, node.payload))
            best = sorted(best, key=lambda x: x[0])[:k]
            axis = depth % len(vector)
            if vector[axis] < node.vector[axis]:
                if node.left_node is not None and isinstance(node.left_node, Node):
                    nodes.append((node.left_node, depth + 1))
                if len(best) < k or abs(vector[axis] - node.vector[axis]) < best[-1][0]:
                    if node.right_node is not None and isinstance(node.right_node, Node):
                        nodes.append((node.right_node, depth + 1))
            else:
                if node.right_node is not None and isinstance(node.right_node, Node):
                    nodes.append((node.right_node, depth + 1))
                if len(best) < k or abs(vector[axis] - node.vector[axis]) < best[-1][0]:
                    if node.left_node is not None and isinstance(node.left_node, Node):
                        nodes.append((node.left_node, depth + 1))
        return best or None


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

        super().__init__(vectorizer, tokenizer)
        self._tree = NaiveKDTree()

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for retriever.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if document is successfully indexed

        In case of corrupt input arguments, False is returned.
        """

        if not (isinstance(documents, list) and documents
                and all(isinstance(doc, str) for doc in documents)):
            return False
        self._documents = []
        self._document_vectors = []
        for doc in documents:
            vector = self._index_document(doc)
            if vector is None:
                return False
            self._documents.append(doc)
            self._document_vectors.append(vector)
        if not self._tree.build(self._document_vectors):
            return False
        return True

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

        if not isinstance(query, str) or not isinstance(n_neighbours, int) or n_neighbours <= 0:
            return None
        tokenized_query = self._tokenizer.tokenize(query)
        if tokenized_query is None:
            return None
        query_vector = self._vectorizer.vectorize(tokenized_query)
        if query_vector is None:
            return None
        nearest_neighbors = self._tree.query(query_vector, n_neighbours)
        if nearest_neighbors is None:
            return None
        relevant_documents = []
        for _, (distance, index) in enumerate(nearest_neighbors):
            if index is not None and index < len(self._documents):
                relevant_documents.append((distance, self._documents[index]))
        return relevant_documents or None

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

        super().__init__(vectorizer, tokenizer)
        self._tree = KDTree()
