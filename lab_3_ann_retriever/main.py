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
    dist = math.sqrt(sum((value - document_vector[i]) ** 2 for i, value in enumerate(query_vector)))
    return dist


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
        for symbol in text:
            if not symbol.isalpha() and symbol != ' ':
                text = text.replace(symbol, ' ')
        without_sp = self._remove_stop_words(text.lower().split())
        if without_sp is None:
            return None
        return without_sp

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if not documents or not isinstance(documents, list) or \
                not all(isinstance(document, str) for document in documents):
            return None
        tokenized_docs = []
        for document in documents:
            doc = self.tokenize(document)
            if doc is None:
                return None
            tokenized_docs.append(doc)
        return tokenized_docs

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens) or \
                not tokens:
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
        if not self._corpus:
            return False
        unique_words = set()
        for doc in self._corpus:
            unique_words.update(set(doc))
        self._vocabulary.extend(sorted(unique_words))
        self._token2ind = {token: i for i, token in enumerate(self._vocabulary)}
        idf = calculate_idf(self._vocabulary, self._corpus)
        if idf is None:
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
        if not isinstance(tokenized_document, list) \
                or not all(isinstance(token, str) for token in tokenized_document) \
                or not tokenized_document:
            return None
        vec = self._calculate_tf_idf(tokenized_document)
        if vec is None:
            return None
        return vec

    def vector2tokens(self, vector: Vector) -> list[str] | None:
        """
        Recreate a tokenized document based on a vector.

        Args:
            vector (Vector): Vector to decode

        Returns:
            list[str] | None: Tokenized document

        In case of corrupt input arguments, None is returned.
        """
        if not vector or not isinstance(vector, tuple) \
                or not all(isinstance(elem, float) for elem in vector) \
                or len(vector) != len(self._token2ind):
            return None
        returned_tokens = []
        for i, value in enumerate(vector):
            if value != 0.0:
                for token, index in self._token2ind.items():
                    if index == i:
                        returned_tokens.append(token)
        return returned_tokens

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
        if not isinstance(document, list) or not all(isinstance(elem, str) for elem in document) \
                or not document:
            return None
        if not self._vocabulary:
            return ()
        tf_idf = [0.0] * len(self._vocabulary)
        tf = calculate_tf(self._vocabulary, document)
        if tf is None:
            return None
        for i, token in enumerate(self._vocabulary):
            if token in document:
                tf_idf[i] = tf.get(token, 0.0) * self._idf_values.get(token, 0.0)
        return tuple(tf_idf)


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
        self._tokenizer = tokenizer
        self._vectorizer = vectorizer
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
        if not isinstance(documents, list) \
                or not all(isinstance(token, str) for token in documents) \
                or not documents:
            return False
        self._documents = documents
        vectors = []
        for doc in self._documents:
            vec = self._index_document(doc)
            if vec is None or None in vec:
                return False
            vectors.append(vec)
        self._document_vectors = vectors
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
        if not isinstance(query, str) or not isinstance(n_neighbours, int):
            return None
        vector_query = self._index_document(query)
        if vector_query is None:
            return None
        most_relevant = self._calculate_knn(vector_query, self._document_vectors, n_neighbours)
        if most_relevant is None or not most_relevant or all(None in v for v in most_relevant):
            return None
        return [(value, self._documents[index]) for index, value in most_relevant]

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
        if query_vector is None or not isinstance(query_vector, tuple) \
                or any(True for vec in self._document_vectors if len(vec) != len(query_vector)):
            return None
        answer = self._calculate_knn(query_vector, self._document_vectors, 1)
        if answer is None or not answer:
            return None
        return self._documents[answer[0][0]]

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
        if query_vector is None or document_vectors is None or n_neighbours is None:
            return None
        if not query_vector or not document_vectors:
            return None
        distances = []
        for index, vec in enumerate(document_vectors):
            distance = calculate_distance(query_vector, vec)
            if distance is None:
                return None
            distances.append((index, distance))
        return sorted(distances, reverse=False, key=lambda t: t[1])[: n_neighbours]

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
        tokenized_doc = self._tokenizer.tokenize(document)
        if tokenized_doc is None:
            return None
        vector = self._vectorizer.vectorize(tokenized_doc)
        if vector is None:
            return None
        return vector

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
        self.left_node = left_node
        self.right_node = right_node
        self.payload = payload

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
        if not vectors or vectors is None or not isinstance(vectors, list) \
                or not all(isinstance(elem, tuple) for elem in vectors):
            return False
        depth = 0
        dim = len(vectors[0])
        all_dimensions = [[vectors, depth, Node(), True]]
        while all_dimensions:
            dimension = all_dimensions.pop()
            dim_vectors = dimension[0]
            dim_parent = dimension[2]
            if not dim_vectors or not dim_parent:
                continue
            axis = dimension[1] % dim
            sorted_vectors = sorted(dim_vectors, key=lambda x: x[axis])
            median_index = len(sorted_vectors) // 2
            median = sorted_vectors[median_index]
            node_median = Node(median, vectors.index(median))
            if dim_parent.payload == -1:
                self._root = node_median
            else:
                if dimension[3]:
                    dim_parent.left_node = node_median
                else:
                    dim_parent.right_node = node_median
            depth += 1
            all_dimensions.append([sorted_vectors[:median_index], depth, node_median, True])
            all_dimensions.append([sorted_vectors[median_index + 1:], depth, node_median, False])
        return True

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
        if not vector or not isinstance(vector, tuple) or not isinstance(k, int):
            return None
        answer = self._find_closest(vector, k)
        if answer is None:
            return None
        return answer

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
        if not vector or not isinstance(vector, tuple) or not isinstance(k, int):
            return None
        nodes = [(self._root, 0)]
        result = []
        while nodes:
            node, depth = nodes.pop()
            if node is None or not isinstance(node.payload, int):
                return None
            if not node.left_node and not node.right_node:
                distance = calculate_distance(vector, node.vector)
                if distance is None:
                    return None
                result.append((distance, node.payload))
            axis = depth % len(vector)
            if vector[axis] <= node.vector[axis]:
                if node.left_node is not None:
                    nodes.append((node.left_node, depth + 1))
            else:
                if node.right_node is not None:
                    nodes.append((node.right_node, depth + 1))
        return result


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
        BasicSearchEngine.__init__(self, vectorizer, tokenizer)
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
        if not isinstance(documents, list) \
                or not all(isinstance(token, str) for token in documents) \
                or not documents:
            return False
        self._documents = documents
        vectors = []
        for doc in self._documents:
            vec = self._index_document(doc)
            if vec is None or None in vec:
                return False
            vectors.append(vec)
        self._document_vectors = vectors
        self._tree.build(self._document_vectors)
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
        if not isinstance(query, str) or not isinstance(n_neighbours, int):
            return None
        vector_query = self._index_document(query)
        if vector_query is None:
            return None
        most_relevant = self._tree.query(vector_query, n_neighbours)
        if most_relevant is None or not most_relevant or all(None in v for v in most_relevant):
            return None
        return [(value, self._documents[index]) for value, index in most_relevant]

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
