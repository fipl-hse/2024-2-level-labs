"""
Lab 3.

Vector search with text retrieving
"""
# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from math import sqrt
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
    if (not isinstance(query_vector, tuple) or
            not isinstance(document_vector, tuple)):
        return None
    if not query_vector or not document_vector:
        return 0.0

    dist = 0.0
    for que_vec, doc_vec in zip(query_vector, document_vector):
        dist += (que_vec - doc_vec) ** 2
    return sqrt(dist)


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
            if not symbol.isalpha():
                text = text.replace(symbol, ' ')
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
        if (not isinstance(documents, list) or
                not all(isinstance(document, str) for document in documents)):
            return None

        tokenized_documents = []
        for document in documents:
            tokenized_doc = self.tokenize(document)
            if not isinstance(tokenized_doc, list):
                return None
            tokenized_documents.append(tokenized_doc)
        return tokenized_documents

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if (not tokens or
                not isinstance(tokens, list) or
                not all(isinstance(token, str) for token in tokens)):
            return None

        clear_list = []
        for token in tokens:
            if token not in self._stop_words:
                clear_list.append(token)
        return clear_list


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

        self._vocabulary = sorted(list(set(token for doc in self._corpus for token in doc)))
        self._idf_values = calculate_idf(self._vocabulary, self._corpus) or {}
        self._token2ind = {word: index for index, word in enumerate(self._vocabulary)}
        return bool(self._vocabulary and self._idf_values and self._token2ind)

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

        return [self._vocabulary[i] for i, value in enumerate(vector) if value > 0]

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
        if (not isinstance(document, list) or
                not document or
                not all(isinstance(token, str) for token in document)):
            return None

        vector = list(0.0 for _ in range(len(self._vocabulary)))
        for token in document:
            if self._token2ind.get(token) is None:
                return None
            tf = calculate_tf(self._vocabulary, document)
            if tf is None:
                return None
            vector[self._token2ind[token]] = tf[token] * self._idf_values[token]
        return Vector(vector)


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
        if (not isinstance(documents, list) or
                not documents or
                not all(isinstance(doc, str) for doc in documents)):
            return False

        for doc in documents:
            indexed_doc = self._index_document(doc)
            if not indexed_doc:
                return False
            self._documents.append(doc)
            self._document_vectors.append(indexed_doc)
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
        if (not isinstance(query, str) or
                not isinstance(n_neighbours, int)):
            return None

        tokenized_query = self._tokenizer.tokenize(query)
        if tokenized_query is None:
            return None
        vectorized_query = self._vectorizer.vectorize(tokenized_query)
        if vectorized_query is None:
            return None
        nearest_neighbours = self._calculate_knn(vectorized_query,
                                                 self._document_vectors, n_neighbours)
        if not nearest_neighbours or any(neighbour is None or neighbour[0] is None or neighbour[1]
                                         is None for neighbour in nearest_neighbours):
            return None

        relevant_documents = []
        for nearest_neighbour in nearest_neighbours:
            if None in nearest_neighbour:
                return None
            relevant_documents.append((nearest_neighbour[1], self._documents[nearest_neighbour[0]]))
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
        if (not isinstance(query_vector, tuple) or
                not len(query_vector) == len(self._document_vectors[0])):
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
        bad_input = (not isinstance(query_vector, (list, tuple)) or
                     not query_vector or
                     not isinstance(document_vectors, list) or
                     not document_vectors or
                     not isinstance(n_neighbours, int) or
                     n_neighbours <= 0)
        if bad_input:
            return None

        neighbours = []
        for document_vector in document_vectors:
            document_distance = calculate_distance(query_vector, document_vector)
            if document_distance is None:
                return None
            neighbours.append((document_vectors.index(document_vector), document_distance))

        neighbours.sort(key=lambda tuple_: tuple_[1])
        return neighbours[:n_neighbours]

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
        vectorized_document = self._vectorizer.vectorize(tokenized_document)
        if vectorized_document is None or not vectorized_document:
            return None
        return vectorized_document

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
        if (not isinstance(vectors, list) or
                not all(isinstance(vector, (list, tuple)) or
                        not all(isinstance(val, float) for val in vector)
                        for vector in vectors) or
                not vectors):
            return False

        space_condition: list[dict] = [
            {
                "vectors": [(vectors[idx], idx) for idx in range(len(vectors))],
                "depth": 0,
                "parent_node": Node(tuple(0.0 for _ in range(len(vectors[0]))), -1),
                "is_left_subspace": True
            }
        ]

        while space_condition:
            current_vectors = space_condition[0]["vectors"]
            depth = space_condition[0]["depth"]
            parent_node = space_condition[0]["parent_node"]
            is_left_sub_dim = space_condition.pop(0)["is_left_subspace"]

            if current_vectors:
                axis = depth % len(current_vectors[0])
                current_vectors.sort(key=lambda vector_with_idx: vector_with_idx[0][axis])
                median_index = len(current_vectors) // 2
                median_node = Node(current_vectors[median_index][0],
                                   current_vectors[median_index][1])

                if parent_node.payload == -1:
                    self._root = median_node
                else:
                    if is_left_sub_dim:
                        parent_node.left_node = median_node
                    else:
                        parent_node.right_node = median_node

                left_subspace = {
                    "vectors": current_vectors[:median_index],
                    "depth": depth + 1,
                    "parent_node": median_node,
                    "is_left_subspace": True
                }
                right_subspace = {
                    "vectors": current_vectors[median_index + 1:],
                    "depth": depth + 1,
                    "parent_node": median_node,
                    "is_left_subspace": False
                }

                space_condition.append(left_subspace)
                space_condition.append(right_subspace)
        if self._root is None:
            return False
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
        if (not isinstance(vector, tuple) or
                not isinstance(k, int) or
                not vector or
                self._root is None or
                not isinstance(self._root, Node)):
            return None

        neighbors_list = []
        nodes = [(self._root, 0)]

        while nodes:
            node = nodes.pop(0)
            current_node, depth = node
            if not isinstance(current_node, Node):
                return None

            if not current_node.left_node and not current_node.right_node:
                dist = calculate_distance(vector, current_node.vector)
                if (not dist or not isinstance(dist, float)
                        or not isinstance(current_node.payload, int)):
                    return None
                neighbors_list.append((dist, current_node.payload))
            else:
                axis = depth % len(current_node.vector)
                if not vector[axis] > current_node.vector[axis]:
                    if isinstance(current_node.left_node, Node):
                        nodes.append((current_node.left_node, depth + 1))
                else:
                    if isinstance(current_node.right_node, Node):
                        nodes.append((current_node.right_node, depth + 1))
        return neighbors_list


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
        if not isinstance(documents, list):
            return False

        self._documents = documents
        document_vectors = []

        for doc in self._documents:
            vector_doc = self._index_document(doc)
            if not isinstance(vector_doc, tuple):
                return False
            document_vectors.append(vector_doc)

        self._document_vectors = document_vectors
        return self._tree.build(self._document_vectors)

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
        if (not isinstance(query, str) or
                not isinstance(n_neighbours, int)):
            return None

        tokenized_query = self._tokenizer.tokenize(query)
        if tokenized_query is None:
            return None
        vectorized_query = self._vectorizer.vectorize(tokenized_query)
        if vectorized_query is None:
            return None

        relevant_vectors = self._tree.query(vectorized_query, n_neighbours)
        if (relevant_vectors is None or
                any(None in vector for vector in relevant_vectors) or
                not relevant_vectors):
            return None

        return [(relevant_vectors[i][0], self._documents[relevant_vectors[i][1]]) for i in
                range(len(relevant_vectors))]

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
