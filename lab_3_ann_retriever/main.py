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
    if query_vector is None or document_vector is None:
        return None
    if not query_vector or not document_vector:
        return 0.0
    return sqrt(sum((query_value - doc_value) ** 2
                    for query_value, doc_value in zip(query_vector, document_vector)))


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
        self.stop_words = stop_words

    def tokenize(self, text: str) -> list[str] | None:
        """
        Tokenize the input text into lowercase words without punctuation, digits and other symbols.

        Args:
            text (str): The input text to tokenize

        Returns:
            list[str] | None: A list of words from the text

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(text, str) or len(text) < 1:
            return None

        for symbol in text:
            if symbol.isalpha() or symbol == " ":
                continue
            text = text.replace(symbol, " ")

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
        if (not isinstance(documents, list) or len(documents) < 1
                or not all(isinstance(doc, str) for doc in documents)):
            return None
        result = []
        for doc in documents:
            new_doc = self.tokenize(doc)
            if new_doc is None:
                return None
            result.append(new_doc)
        return result

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if tokens is None or not isinstance(tokens, list) or not isinstance(self.stop_words, list):
            return None
        for token in tokens:
            if not isinstance(token, str):
                return None
        for word in self.stop_words:
            if not isinstance(word, str):
                return None
        if len(tokens) == 0 or len(self.stop_words) == 0:
            return None

        return list(filter(lambda x: x not in self.stop_words, tokens))


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
        if (not self._corpus or not isinstance(self._corpus, list)
                or not all(isinstance(doc, list) for doc in self._corpus)):
            return False

        self._vocabulary = sorted(set(token for doc in self._corpus for token in doc))
        self._token2ind = {token: index for index, token in enumerate(self._vocabulary)}
        idf_values = calculate_idf(self._vocabulary, self._corpus)
        if idf_values is None:
            return False
        self._idf_values = idf_values

        if self._idf_values is None:
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
        if (not isinstance(tokenized_document, list) or not tokenized_document or
                not all(isinstance(token, str) for token in tokenized_document)):
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
        if not vector or len(vector) != len(self._token2ind):
            return None

        tokenized_doc = []
        for i, num in enumerate(vector):
            if num == 0.0:
                continue
            for token, ind in self._token2ind.items():
                if i == ind:
                    tokenized_doc.append(token)

        return tokenized_doc

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

        vector = list(0.0 for _ in range(len(self._vocabulary)))
        tf_values = calculate_tf(self._vocabulary, document)
        for index, word in enumerate(self._vocabulary):
            vector[index] = tf_values.get(word, 0) * self._idf_values.get(word, 0)

        return tuple(vector)


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
        if (not isinstance(documents, list) or not documents
                or not all(isinstance(doc, str) for doc in documents)):
            return False
        self._documents = documents

        for doc in documents:
            indexed_doc = self._index_document(doc)
            if indexed_doc is None:
                return False
            self._document_vectors.append(indexed_doc)

        if (self._document_vectors is None or not self._document_vectors
                or len(self._document_vectors) != len(documents)):
            return False

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
        if not query_vector or not isinstance(query_vector, tuple):
            return None

        for vector in self._document_vectors:
            if len(vector) < len(query_vector):
                return None
        knn_list = self._calculate_knn(query_vector, self._document_vectors, 1)
        if not knn_list or knn_list is None:
            return None
        doc_ind = knn_list[0][0]
        if not doc_ind:
            return None

        return self._documents[doc_ind]

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
        bad_input = (not document_vectors or not query_vector or not n_neighbours
                     or not isinstance(n_neighbours, int) or isinstance(n_neighbours, bool)
                     or n_neighbours > len(document_vectors)
                     or not all(isinstance(doc, tuple) for doc in document_vectors))
        if bad_input:
            return None

        distances = []
        for doc_vector in document_vectors:
            distance = calculate_distance(query_vector, doc_vector)
            if not isinstance(distance, float):
                return None
            distances.append((document_vectors.index(doc_vector), distance))
        distances_sorted = sorted(distances, key=lambda x: x[1])

        return distances_sorted[:n_neighbours]

    def _index_document(self, document: str) -> Vector | None:
        """
        Index document.

        Args:
            document (str): Document to index

        Returns:
            Vector | None: Returns document vector

        In case of corrupt input arguments, None is returned.
        """
        if not document or not isinstance(document, str):
            return None

        tokenized_doc = self._tokenizer.tokenize(document)
        if tokenized_doc is None:
            return None

        vectorized_doc = self._vectorizer.vectorize(tokenized_doc)
        if vectorized_doc is None:
            return None

        return vectorized_doc

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
        if (not vectors or
                not all(isinstance(vector, tuple) and len(vector) > 0 for vector in vectors)):
            return False

        space_state = [(vectors, 0, Node((), -1), True)]

        while space_state:
            current_vectors, depth, parent_node, left_dimension = space_state.pop()

            if not current_vectors:
                continue

            axis = depth % len(vectors[0])
            sorted_vectors = sorted(current_vectors, key=lambda vector: vector[axis])
            median_index = len(sorted_vectors) // 2
            median_vector = sorted_vectors[median_index]
            median_node = Node(vector=sorted_vectors[median_index],
                               payload=vectors.index(median_vector))

            if parent_node.payload == -1:
                self._root = median_node
            else:
                if left_dimension:
                    parent_node.left_node = median_node
                else:
                    parent_node.right_node = median_node

            left_vectors = sorted_vectors[:median_index]
            right_vectors = sorted_vectors[median_index + 1:]
            depth += 1

            space_state.append((left_vectors, depth, median_node, True))
            space_state.append((right_vectors, depth, median_node, False))

        if not self._root:
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
        if not isinstance(vector, (tuple, list)) or not isinstance(k, int) or not vector or not k:
            return None

        return self._find_closest(vector, k) or None

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
        if not isinstance(vector, tuple) or not isinstance(k, int) or not vector:
            return None

        subspaces = [(self._root, 0)]
        while subspaces:
            node, depth = subspaces.pop(0)
            if node is None:
                return None
            if node.left_node is None and node.right_node is None:
                distance = calculate_distance(vector, node.vector)
                if distance is None:
                    return None
                return [(distance, node.payload)]
            axis = depth % len(node.vector)
            current_depth = depth + 1
            if vector[axis] <= node.vector[axis]:
                subspaces.append((node.left_node, current_depth))
            subspaces.append((node.right_node, current_depth))

        return None


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
        if (not isinstance(documents, list) or not documents
                or not all(isinstance(doc, str) for doc in documents)):
            return False

        self._documents = documents
        if not self._documents:
            return False

        for doc in documents:
            indexed_doc = self._index_document(doc)
            if indexed_doc is None:
                return False
            self._document_vectors.append(indexed_doc)
        self._tree.build(self._document_vectors)

        if (self._document_vectors is None or not self._document_vectors
                or len(self._document_vectors) != len(documents)
                or not self._tree.build(self._document_vectors)):
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
        if not isinstance(query, str) or not isinstance(n_neighbours, int) or n_neighbours < 1:
            return None
        tokenized_query = self._tokenizer.tokenize(query)
        if tokenized_query is None:
            return None
        query_vector = self._vectorizer.vectorize(tokenized_query)
        if query_vector is None:
            return None
        nearest_neighbours = self._tree.query(query_vector)
        if not nearest_neighbours:
            return None
        relevant_documents = []

        for score, index in nearest_neighbours:
            if index is None or index < len(self._documents) or not isinstance(index, int):
                return None
            relevant_documents.append((score, self._documents[index]))

        if not isinstance(relevant_documents, list) or relevant_documents == []:
            return None

        return relevant_documents

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
