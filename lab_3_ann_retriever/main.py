"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from re import findall
from typing import Protocol
from lab_2_retrieval_w_bm25.main import (calculate_idf, calculate_tf)


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

    return sum((query - doc) ** 2 for query, doc in zip(query_vector, document_vector)) ** 0.5


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
        if not isinstance(text, str) or not text:
            return None

        return self._remove_stop_words(findall(r'[a-zа-яё]+', text.lower()))

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(documents, list) or not documents \
                or not all(isinstance(doc, str) for doc in documents):
            return None

        return [tok_doc for doc in documents if (tok_doc := self.tokenize(doc)) is not None] or None

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(tokens, list) or not tokens \
                or not all(isinstance(token, str) for token in tokens):
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
        self._vocabulary = []
        self._idf_values = {}
        self._token2ind = {}

    def build(self) -> bool:
        """
        Build vocabulary with tokenized_documents.

        Returns:
            bool: True if built successfully, False in other case
        """
        if not self._corpus:
            return False

        self._vocabulary = sorted(list({term for doc in self._corpus for term in doc}))
        idf = calculate_idf(self._vocabulary, self._corpus)
        if not idf:
            return False
        self._idf_values = idf
        self._token2ind = {term: index for index, term in enumerate(self._vocabulary)}
        if not self._idf_values and self._vocabulary or not self._token2ind:
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

        return self._calculate_tf_idf(tokenized_document) or ()

    def vector2tokens(self, vector: Vector) -> list[str] | None:
        """
        Recreate a tokenized document based on a vector.

        Args:
            vector (Vector): Vector to decode

        Returns:
            list[str] | None: Tokenized document

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(vector, tuple) or not vector \
                or len(vector) != len(self._token2ind):
            return None

        tokenized_doc = []
        for index, score in enumerate(vector):
            if not score:
                continue
            for term, i in self._token2ind.items():
                if index == i:
                    tokenized_doc.append(term)
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
        if not isinstance(document, list) or not document \
                or not all(isinstance(token, str) for token in document):
            return None
        tf = calculate_tf(self._vocabulary, document)
        if not self._idf_values or not tf:
            return None
        tf_idf = {term: tf[term] * self._idf_values[term] for term in self._vocabulary}
        return tuple(tf_idf.values())


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
        if not isinstance(documents, list) or not documents \
                or not all(isinstance(term, str) for term in documents):
            return False

        self._documents = documents
        self._document_vectors = [vect for doc in documents if (vect := self._index_document(doc))]
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
        if not isinstance(query, str) or not query or not isinstance(n_neighbours, int) \
                or n_neighbours <= 0:
            return None

        tokenized_query = self._tokenizer.tokenize(query)
        if not tokenized_query:
            return None
        query_vector = self._vectorizer.vectorize(tokenized_query)
        if not query_vector:
            return None
        knn = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if not knn:
            return None

        return [(score, self._documents[index]) for index, score in knn
                if isinstance(index, int) and score is not None] or None

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
        if not isinstance(query_vector, tuple) or not query_vector \
                or not all(len(doc_vector) == len(query_vector)
                           for doc_vector in self._document_vectors):
            return None

        knn = self._calculate_knn(query_vector, self._document_vectors, 1)

        return None if not knn else self._documents[knn[0][0]]

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
        if not isinstance(query_vector, (tuple, list)) or not query_vector \
                or not isinstance(document_vectors, list) or not document_vectors \
                or not isinstance(n_neighbours, int):
            return None

        distances = [(document_vectors.index(doc_vect), distance) for doc_vect in document_vectors
                     if (distance := calculate_distance(query_vector, doc_vect)) is not None]
        return sorted(distances, key=lambda item: item[1])[:n_neighbours]

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
        if not tokenized_doc:
            return None

        return self._vectorizer.vectorize(tokenized_doc)

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
        if not isinstance(vectors, list) or not vectors \
                or not all(isinstance(vector, (tuple, list)) for vector in vectors):
            return False

        space = [
            {
             'vectors': [(vector, index) for index, vector in enumerate(vectors)],
             'depth': 0,
             'parent node': Node(),
             'current dimension': True
            }
        ]
        dimensions = len(vectors[0])

        while space:
            current_vectors = space[0]['vectors']
            depth = space[0]['depth']
            parent_node = space[0]['parent node']
            current_dimension = space.pop(0)['current dimension']
            if not isinstance(current_vectors, list) or not isinstance(depth, int):
                return False
            if not current_vectors:
                continue
            axis = depth % dimensions
            current_vectors.sort(key=lambda vector: vector[0][axis])
            median_index = len(current_vectors) // 2
            median_node = Node(current_vectors[median_index][0], current_vectors[median_index][1])

            if parent_node.payload == -1:
                self._root = median_node
            else:
                if not current_dimension:
                    parent_node.right_node = median_node
                else:
                    parent_node.left_node = median_node

            space.append(
                {
                    'vectors': current_vectors[:median_index],
                    'depth': depth + 1,
                    'parent node': median_node,
                    'current dimension': True
                }
            )
            space.append(
                {
                    'vectors': current_vectors[median_index + 1:],
                    'depth': depth + 1,
                    'parent node': median_node,
                    'current dimension': False
                }
            )

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
        if not isinstance(vector, (tuple, list)) or not vector or not isinstance(k, int) or not k:
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
        if not isinstance(vector, tuple) or not vector or not isinstance(k, int):
            return None

        subspaces = [(self._root, 0)]
        dimensions = len(vector)
        nearest_neighbours = []
        while subspaces:
            item = subspaces.pop(0)
            if not item[0]:
                continue
            if not item[0].left_node and not item[0].right_node:
                distance = calculate_distance(vector, item[0].vector)
                if not isinstance(distance, float):
                    continue
                nearest_neighbours.append((distance, item[0].payload))
            axis = item[1] % dimensions
            if vector[axis] <= item[0].vector[axis]:
                if item[0].left_node is not None:
                    subspaces.append((item[0].left_node, item[1] + 1))
            else:
                if item[0].right_node is not None:
                    subspaces.append((item[0].right_node, item[1] + 1))
        return sorted(nearest_neighbours, key=lambda x: x[0])[:k] or None


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
        if not isinstance(documents, (list, tuple)) or not documents \
                or not all(isinstance(term, str) for term in documents):
            return False

        self._documents = documents
        self._document_vectors = [vect for doc in documents if (vect := self._index_document(doc))]
        self._tree.build(self._document_vectors)
        if not self._documents or not self._document_vectors or not self._tree:
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
        if not isinstance(query, str) or not query or not isinstance(n_neighbours, int) \
                or n_neighbours <= 0:
            return None

        tokenized_query = self._tokenizer.tokenize(query)
        if not tokenized_query:
            return None
        query_vector = self._vectorizer.vectorize(tokenized_query)
        if not query_vector:
            return None
        retrieved_answer = self._tree.query(query_vector, n_neighbours)
        if not retrieved_answer:
            return None

        return [(score, self._documents[index])
                for score, index in retrieved_answer if score is not None] or None

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
