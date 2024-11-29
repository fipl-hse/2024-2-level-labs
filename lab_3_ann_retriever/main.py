"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from typing import Protocol

from lab_2_retrieval_w_bm25.main import calculate_idf

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

    summary = 0.0
    for i, value in enumerate(document_vector):
        summary += (query_vector[i] - value) ** 2
    euclidean_distance = summary ** 0.5

    if not isinstance(euclidean_distance, float):
        return None

    return euclidean_distance


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
        punctuation = ("!'\"#$%&()*+,-./:–;—<=>?@[]^_`{|}~1234567890"
                       r'\"'
                       "")
        for p in punctuation:
            if p in text:
                text = text.replace(p, ' ')
        tokens = text.lower().split()

        return self._remove_stop_words(tokens)

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(documents, list) or not documents:
            return None
        for text in documents:
            if not isinstance(text, str) or len(text) == 0 or self.tokenize(text) is None:
                return None

        tokenized_docs = []
        for text in documents:
            processed_text = self.tokenize(text)
            if processed_text is None:
                return None
            tokenized_docs.append(processed_text)
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
        if not isinstance(tokens, list) or not tokens:
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
        vocabulary = set()
        if not isinstance(self._corpus, list):
            return False
        for doc in self._corpus:
            if not isinstance(doc, list) or not doc:
                return False
            vocabulary |= set(doc)
        sorted_vocab = sorted(vocabulary)
        self._vocabulary = sorted_vocab

        idf = calculate_idf(self._vocabulary, self._corpus)
        if idf is None or not isinstance(idf, dict):
            return False
        self._idf_values = idf
        if self._idf_values is None:
            return False

        for word in self._vocabulary:
            self._token2ind[word] = self._vocabulary.index(word)

        if None in self._corpus or not self._vocabulary or not self._idf_values:
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
        if not isinstance(tokenized_document, list) or not tokenized_document:
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
        if len(vector) != len(self._vocabulary):
            return None
        tokens = []
        for word in self._vocabulary:
            if vector[self._token2ind[word]] != 0.0:
                tokens.append(word)
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
        if not isinstance(document, list) or not all(isinstance(token, str) for token in document) \
                or not document:
            return None

        vector = [0.0] * len(self._vocabulary)
        for token in document:
            if token in self._vocabulary:
                tf = {}
                for word in set(self._vocabulary) | set(document):
                    tf[word] = document.count(word) / len(document)
                if not isinstance(tf, dict):
                    return None
                if token in tf and token in self._idf_values:
                    tf_idf = tf[token] * self._idf_values[token]
                    vector[self._token2ind[token]] = tf_idf
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
        if not isinstance(documents, list) or not documents:
            return False
        self._documents = documents
        for text in documents:
            if not isinstance(text, str):
                return False
            vector = self._index_document(text)
            if not isinstance(vector, tuple):
                return False
            self._document_vectors.append(vector)
        if not self._document_vectors:
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
        if not isinstance(query, str) or not query \
                or not isinstance(n_neighbours, int) or n_neighbours <= 0:
            return None

        tokens = self._tokenizer.tokenize(query)
        if not isinstance(tokens, list):
            return None
        query_vector = self._vectorizer.vectorize(tokens)
        if not isinstance(query_vector, tuple):
            return None
        closest_neighbours = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if closest_neighbours is None or not closest_neighbours or len(closest_neighbours) == 0 or \
                not all(isinstance(index, int) or isinstance(distance, float)
                        for index, distance in closest_neighbours):
            return None
        for item in closest_neighbours:
            if item[0] is None or item[1] is None:
                return None

        result = []
        for i, distance in closest_neighbours:
            result.append((distance, self._documents[i]))
        return result

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
        if not isinstance(query_vector, tuple) or \
                len(query_vector) != len(self._document_vectors[0]):
            return None

        knn = self._calculate_knn(query_vector, self._document_vectors, 1)
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
        if query_vector is None:
            return None
        if not isinstance(document_vectors, list) or len(document_vectors) == 0:
            return None
        if n_neighbours <= 0 or not isinstance(n_neighbours, int):
            return None

        distances = []
        for vector in document_vectors:
            distances.append(calculate_distance(query_vector, vector))
            if distances is None or not distances or isinstance(distances, tuple):
                return None
            for elem in enumerate(distances):
                if not isinstance(elem[1], float):
                    return None
        neighbours = sorted(enumerate(distances), key=lambda x: x[1])
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
        tokenized_doc = self._tokenizer.tokenize(document)
        if not isinstance(tokenized_doc, list):
            return None
        doc_vector = self._vectorizer.vectorize(tokenized_doc)
        if not isinstance(doc_vector, tuple):
            return None
        return doc_vector

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
        if not isinstance(vectors, list) or not vectors:
            return False

        states_info = [{}]
        states_info = [{
            'vectors': [(vector, index) for index, vector in enumerate(vectors)],
            'depth': 0,
            'parent': Node(tuple([0.0] * len(vectors[0])), -1),
            'is_left': True
        }]
        while states_info:
            current_vectors, depth, parent, is_left = states_info.pop(0).values()
            if current_vectors:
                axis = depth % len(current_vectors[0])
                current_vectors.sort(key=lambda vector: vector[0][axis])
                median_index = len(current_vectors) // 2
                median_node = Node(current_vectors[median_index][0],
                                   current_vectors[median_index][1])
                if parent.payload != -1 and is_left:
                    parent.left_node = median_node
                elif parent.payload == -1:
                    self._root = median_node
                else:
                    parent.right_node = median_node
                states_info.append(
                    {
                        'vectors': current_vectors[:median_index],
                        'depth': depth + 1,
                        'parent': median_node,
                        'is_left': True
                    }
                )
                states_info.append(
                    {
                        'vectors': current_vectors[median_index + 1:],
                        'depth': depth + 1,
                        'parent': median_node,
                        'is_left': False
                    }
                )
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
        if not isinstance(vector, tuple) or not vector:
            return None
        if len(vector) == 0:
            return None
        if not isinstance(k, int) or not k:
            return None

        return self._find_closest(vector)

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
        if not isinstance(vector, tuple) or not vector or not isinstance(k, int):
            return None

        node_depth_list = [
            (self._root, 0)
        ]

        while True:
            node, depth = node_depth_list.pop(0)
            if node.left_node is None and node.right_node is None:
                distance = calculate_distance(vector, node.vector)
                if distance is None:
                    return None
                return [(distance, node.payload)]
            axis = depth % len(node.vector)
            if vector[axis] <= node.vector[axis]:
                node_depth_list.append((node.left_node, depth + 1))
            else:
                node_depth_list.append((node.right_node, depth + 1))


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
        super().__init__(vectorizer,tokenizer)
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
        if not super().index_documents(documents):
            return False
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
        if not isinstance(query, str) or not query \
                or not isinstance(n_neighbours, int) or n_neighbours != 1:
            return None
        if len(query) == 0:
            return None

        query_vector = self._index_document(query)
        if query_vector is None:
            return None

        distances_indices_list = self._tree.query(query_vector)
        if distances_indices_list is None or not distances_indices_list or not \
                all(isinstance(distance, float) or isinstance(index, int)
                    for distance, index in distances_indices_list):
            return None

        result = []
        for dist, doc in distances_indices_list:
            result.append((dist, self._documents[doc]))

        return result[:n_neighbours]

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
