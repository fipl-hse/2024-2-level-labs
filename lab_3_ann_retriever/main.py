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

    return sqrt(sum((query_value - document_value) ** 2
                    for query_value, document_value in zip(query_vector, document_vector)))


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

        for letter in text:
            if not letter.isalpha() and letter != ' ':
                text = text.replace(letter, ' ')
        clean_text = text.lower().split()

        return self._remove_stop_words(clean_text)

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if (not isinstance(documents, list)
                or not all(isinstance(document, str) for document in documents)
                or not documents):
            return None

        list_of_tok_docs = []
        for text in documents:
            doc = self.tokenize(text)
            if not doc:
                return None
            list_of_tok_docs.append(doc)

        return list_of_tok_docs

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if (not isinstance(tokens, list)
                or not all(isinstance(one_token, str) for one_token in tokens)
                or not tokens):
            return None

        return [word for word in tokens if word not in self._stop_words]


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
        self._vocabulary = []
        self._token2ind = {}
        self._idf_values = {}

    def build(self) -> bool:
        """
        Build vocabulary with tokenized_documents.

        Returns:
            bool: True if built successfully, False in other case
        """
        if (not self._corpus
                or not all(isinstance(sublist, list) for sublist in self._corpus)):
            return False

        vocabulary = set()
        for sublist in self._corpus:
            if sublist is None:
                return False
            vocabulary.update(set(sublist))
        self._vocabulary = sorted(list(vocabulary))

        if not self._vocabulary:
            return False
        idf_values = calculate_idf(self._vocabulary, self._corpus)
        if idf_values is None:
            return False
        self._idf_values = idf_values

        for word in self._vocabulary:
            self._token2ind[word] = self._vocabulary.index(word)

        if (self._token2ind
                and None not in self._idf_values and None not in self._token2ind):
            return True
        return False

    def vectorize(self, tokenized_document: list[str]) -> Vector | None:
        """
        Create a vector for tokenized document.

        Args:
            tokenized_document (list[str]): Tokenized document to vectorize

        Returns:
            Vector | None: TF-IDF vector for document

        In case of corrupt input arguments, None is returned.
        """
        if (not isinstance(tokenized_document, list)
                or not all(isinstance(token, str) for token in tokenized_document)
                or not tokenized_document):
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
        if len(vector) != len(self._idf_values):
            return None

        result = []
        for every_word in self._vocabulary:
            if vector[self._token2ind[every_word]] != 0.0:
                result.append(every_word)

        return result

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
        if (not isinstance(document, list)
                or not all(isinstance(token, str) for token in document)
                or not document):
            return None

        vector = [0.0 for _ in self._vocabulary]

        tf = calculate_tf(self._vocabulary, document)
        for index, word in enumerate(self._vocabulary):
            vector[index] = tf.get(word, 0) * self._idf_values.get(word, 0)

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
        self._document_vectors = []
        self._documents = []

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for engine.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if documents are successfully indexed

        In case of corrupt input arguments, False is returned.
        """
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents) \
                or not documents:
            return False

        self._document_vectors = [self._index_document(doc) for doc in documents]
        self._documents = documents

        if self._document_vectors and None not in self._document_vectors:
            return True
        return False

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
        if (n_neighbours <= 0 or not isinstance(query, str)
                or not query or not isinstance(n_neighbours, int)):
            return None

        query_vectorized = self._index_document(query)
        if query_vectorized is None:
            return None

        self.index_documents(self._documents)
        knn_list_of_tuples = self._calculate_knn(query_vectorized,
                                                 self._document_vectors, n_neighbours)
        if knn_list_of_tuples is None or not knn_list_of_tuples:
            return None

        relevant_docs = []
        for index, float_as_value in knn_list_of_tuples:
            if float_as_value is None:
                return None
            relevant_docs.append((float_as_value, self._documents[index]))
        if relevant_docs is None:
            return None
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
        if (not isinstance(query_vector, (list, tuple))
                or not query_vector
                or not self._document_vectors):
            return None
        if (not isinstance(query_vector, tuple)
                or len(query_vector) != len(self._document_vectors[0])):
            return None

        knn_list_of_tuples = self._calculate_knn(query_vector, self._document_vectors, 1)
        if knn_list_of_tuples is None or not knn_list_of_tuples:
            return None
        return self._documents[knn_list_of_tuples[0][0]]

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
        if (not isinstance(document_vectors, list) or not isinstance(n_neighbours, int)
                or not query_vector or not document_vectors):
            return None

        distances = []
        for value_in_tuple in document_vectors:
            distance = calculate_distance(query_vector, value_in_tuple)
            if distance is None:
                return None
            distances.append((document_vectors.index(value_in_tuple), distance))
        distances = sorted(distances, key=lambda x: x[1], reverse=False)

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
        if not isinstance(document, str) or not document:
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
        if not vectors or not isinstance(vectors, list):
            return False

        depth = 0
        states_of_space = [(vectors, depth, Node(), True)]

        while states_of_space:
            space_vector, depth, parent_node, left_node = states_of_space.pop()
            if not space_vector:
                continue

            dimensions = len(vectors[0])
            axis = depth % dimensions
            depth += 1
            sorted_vectors = sorted(space_vector, key=lambda vector: vector[axis])
            median_index = len(sorted_vectors) // 2
            median_dot = sorted_vectors[median_index]
            new_space_node = Node(median_dot, vectors.index(median_dot))

            if parent_node.payload == -1:
                self._root = new_space_node
            else:
                if left_node:
                    parent_node.left_node = new_space_node
                else:
                    parent_node.right_node = new_space_node

            left_vectors = sorted_vectors[:median_index]
            right_vectors = sorted_vectors[median_index + 1:]

            states_of_space.append((left_vectors, depth, new_space_node, True))
            states_of_space.append((right_vectors, depth, new_space_node, False))

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
        if not isinstance(k, int) or not vector:
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
        if not vector or not isinstance(vector, tuple) or not isinstance(k, int):
            return None

        pairs = [(self._root, 0)]
        while pairs:
            node, depth = pairs.pop(0)
            if node is None or not isinstance(node.payload, int):
                return None

            if node.left_node is None and node.right_node is None:
                distance = calculate_distance(vector, node.vector)
                if distance is None:
                    return None
                return [(distance, node.payload)]

            axis = depth % len(vector)
            if vector[axis] < node.vector[axis]:
                if node.left_node is not None:
                    pairs.append((node.left_node, depth + 1))
            else:
                if node.right_node is not None:
                    pairs.append((node.right_node, depth + 1))
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
        if (not isinstance(documents, list) or not documents
                or not all(isinstance(elem, str) for elem in documents)):
            return False

        if super().index_documents(documents) is False:
            return False
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
        if (not isinstance(query, str) or not query
                or not isinstance(n_neighbours, int) or n_neighbours <= 0):
            return None

        query_vector = self._index_document(query)
        if not query_vector or query_vector is None:
            return None

        distances = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if not distances or distances is None:
            return None

        docs_plus_dist = []
        for distance in distances:
            if not isinstance(distance, tuple) or not isinstance(distance[0], int):
                return None
            docs_plus_dist.append((distance[1], self._documents[distance[0]]))
        return docs_plus_dist

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
        self._tree = KDTree()
        super().__init__(vectorizer, tokenizer)
