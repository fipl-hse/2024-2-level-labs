"""
Lab 3.

Vector search with text retrieving
"""

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
                document_vector) == 0)
    ):
        return None
    if not query_vector or not document_vector:
        return 0.0
    distance: float = 0.0
    for idx, query_cor in enumerate(query_vector):
        distance += (query_cor - document_vector[idx]) ** 2
    return sqrt(distance)


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
        self._stop_words = stop_words.copy()

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
        if not (documents and isinstance(documents, list) and all(
                isinstance(document, str) for document in documents)):
            return None
        tokenized_documents = []
        for document in documents:
            tokenized_document = self.tokenize(document)
            if tokenized_document is None:
                return None
            tokenized_documents.append(tokenized_document)
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
        if not (isinstance(tokens, list) and tokens and all(
                isinstance(token, str) for token in tokens)):
            return None
        return list(filter(lambda word: word not in self._stop_words, tokens))


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
        if not (isinstance(corpus, list) and all(
                isinstance(tokens, list) and all(isinstance(token, str) for token in tokens) for
                tokens in corpus)):
            self._corpus = []
        else:
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
        for tokenized_document in self._corpus:
            for token in tokenized_document:
                if token not in self._vocabulary:
                    self._vocabulary.append(token)

        self._vocabulary.sort()
        for token in self._vocabulary:
            self._token2ind[token] = self._vocabulary.index(token)

        idf: dict[str, float] | None = calculate_idf(self._vocabulary, self._corpus)
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
                isinstance(token, str) for token in tokenized_document)):
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
        tokenized_doc = []
        for token in self._vocabulary:
            if vector[self._token2ind[token]] != 0:
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
        if not (isinstance(document, list) and document and all(
                isinstance(token, str) for token in document)):
            return None

        vector = list(0.0 for _ in range(len(self._vocabulary)))
        for token in document:
            if self._token2ind.get(token) is not None:
                tf: dict[str, float] | None = calculate_tf(self._vocabulary, document)
                if tf is not None:
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
        if not (isinstance(documents, list) and documents and all(
                isinstance(doc, str) for doc in documents)):
            return False
        self._documents = documents

        for document in documents:
            indexed_document: tuple[float, ...] | None = self._index_document(document)
            if indexed_document is not None:
                self._document_vectors.append(indexed_document)
            else:
                return False

        if None in self._document_vectors:
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
        tokenized_query: list[str] | None = self._tokenizer.tokenize(query)
        if tokenized_query is None:
            return None
        vector_query: tuple[float, ...] | None = self._vectorizer.vectorize(tokenized_query)
        if vector_query is None:
            return None
        n_distances = self._calculate_knn(vector_query, self._document_vectors, n_neighbours)
        if not n_distances:
            return None

        relevant_docs = []

        for distance in n_distances:
            relevant_docs.append((distance[1], self._documents[distance[0]]))
        return relevant_docs

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
        if not (isinstance(query_vector, tuple) and all(
                isinstance(cor, float) for cor in query_vector)):
            return None
        knn: list[tuple[int, float]] | None = self._calculate_knn(query_vector,
                                                                  self._document_vectors, 1)
        if knn is None:
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
                and all(isinstance(vector, (tuple, list))
                        and all(isinstance(cor, float) for cor in vector) for vector in
                        document_vectors)
                and isinstance(n_neighbours, int)):
            return None
        distances: list[tuple[int, float]] = []
        for document_vector in document_vectors:
            distance: float | None = calculate_distance(query_vector, document_vector)
            if distance is None:
                return None
            distances.append((document_vectors.index(document_vector), distance))
        distances.sort(key=lambda distance: distance[1])
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
        tokenized_document: list[str] | None = self._tokenizer.tokenize(document)
        if tokenized_document is None:
            return None
        return self._vectorizer.vectorize(tokenized_document)

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
        if not (isinstance(vectors, list) and all(isinstance(vector, (list, tuple)) and all(
                isinstance(cor, float) for cor in vector) for vector in vectors) and vectors):
            return False
        dimensions_info: list[dict] = [
            {
                "vectors": [(vectors[idx], idx) for idx in range(len(vectors))],
                "depth": 0,
                "parentNode": Node(tuple(0.0 for _ in range(len(vectors[0]))), -1),
                "isLeftSubDim": True
            }
        ]

        while dimensions_info:
            cur_vectors: list[tuple[Vector, int]] = dimensions_info[0]["vectors"]
            depth: int = dimensions_info[0]["depth"]
            parent_node: Node = dimensions_info[0]["parentNode"]
            is_left_sub_dim: bool = dimensions_info.pop(0)["isLeftSubDim"]
            if cur_vectors:
                axis = depth % len(cur_vectors[0])
                cur_vectors.sort(key=lambda vector_with_idx: vector_with_idx[0][axis])
                median_index = len(cur_vectors) // 2
                median_node = Node(cur_vectors[median_index][0], cur_vectors[median_index][1])

                if parent_node.payload == -1:
                    self._root = median_node
                else:
                    if is_left_sub_dim:
                        parent_node.left_node = median_node
                    else:
                        parent_node.right_node = median_node
                left_dim = {
                    "vectors": cur_vectors[:median_index],
                    "depth": depth + 1,
                    "parentNode": median_node,
                    "isLeftSubDim": True
                }
                right_dim = left_dim.copy()
                right_dim["vectors"] = cur_vectors[median_index + 1:]
                right_dim["isLeftSubDim"] = False
                dimensions_info.append(left_dim)
                dimensions_info.append(right_dim)
            else:
                continue
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
        if not (isinstance(vector, (list, tuple)) and all(
                isinstance(cor, float) for cor in vector) and isinstance(k, int)
                and vector and self._root is not None and isinstance(self._root, Node)):
            return None
        nodes: list[tuple[Node, int]] = [(self._root, 0)]
        while nodes:
            cur_node: tuple[Node, int] = nodes.pop(0)
            if cur_node[0].left_node is None and cur_node[0].right_node is None:
                distance: float | None = calculate_distance(vector, cur_node[0].vector)
                if distance is not None:
                    return [(distance, cur_node[0].payload)]
            axis = cur_node[1] % len(cur_node[0].vector)
            if cur_node[0].vector[axis] > vector[axis]:
                if cur_node[0].left_node is not None and isinstance(cur_node[0].left_node, Node):
                    nodes.append((cur_node[0].left_node, cur_node[1] + 1))
            else:
                if cur_node[0].right_node is not None and isinstance(cur_node[0].right_node, Node):
                    nodes.append((cur_node[0].right_node, cur_node[1] + 1))
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
        if not (isinstance(vector, (tuple, list)) and all(isinstance(cor, float) for cor in vector)
                and isinstance(k, int) and vector and isinstance(self._root, Node)):
            return None
        best: list[tuple[float, int]] = []
        cur_nodes_depth: list[tuple[Node, int]] = [(self._root, 0)]

        while cur_nodes_depth:
            cur_node_depth = cur_nodes_depth.pop(0)
            distance = calculate_distance(vector, cur_node_depth[0].vector)
            if distance is None:
                return None
            if len(best) < k or distance < max(best, key=lambda best_p: best_p[0])[0]:
                best.append((distance, cur_node_depth[0].payload))
                if len(best) > k:
                    del best[best.index(max(best, key=lambda best_p: best_p[0]))]
            axis = cur_node_depth[1] % len(vector)
            if vector[axis] < cur_node_depth[0].vector[axis]:
                closest_node = cur_node_depth[0].left_node
                far_node = cur_node_depth[0].right_node
            else:
                closest_node = cur_node_depth[0].right_node
                far_node = cur_node_depth[0].left_node
            if isinstance(closest_node, Node):
                cur_nodes_depth.append((closest_node, cur_node_depth[1] + 1))
            if (vector[axis] - cur_node_depth[0].vector[axis]) < \
                    max(best, key=lambda best_p: best_p[0])[0] and isinstance(far_node, Node):
                cur_nodes_depth.append((far_node, cur_node_depth[1] + 1))
        return best


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
        if not (isinstance(documents, list) and documents and all(
                isinstance(doc, str) for doc in documents)):
            return False
        self._documents = documents

        for document in documents:
            indexed_document: tuple[float, ...] | None = self._index_document(document)
            if indexed_document is not None:
                self._document_vectors.append(indexed_document)
            else:
                return False

        if None in self._document_vectors or not self._tree.build(self._document_vectors):
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
        if not (isinstance(query, str) and isinstance(n_neighbours, int)):
            return None
        query_tokens = self._tokenizer.tokenize(query)
        if query_tokens is None:
            return None
        query_vector = self._vectorizer.vectorize(query_tokens)
        if query_vector is None:
            return None
        relevant_vector = self._tree.query(query_vector)
        if relevant_vector is None:
            return None
        return [(relevant_vector[i][0], self._documents[relevant_vector[i][1]]) for i in
                range(len(relevant_vector))]

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
        super.__init__()
