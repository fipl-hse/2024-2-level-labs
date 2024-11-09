"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from typing import Protocol
import math
import json
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
    if not query_vector or not document_vector:
        return None
    if len(query_vector) != len(document_vector):
        return None
    sum_ = 0.0
    for ind, value in enumerate(query_vector):
        sum_ += (value - document_vector[ind]) ** 2
    return sum_ ** 0.5


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """
    elements = {ind: value for ind, value in enumerate(vector) if value}
    return {'len': len(vector), 'elements': elements}


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if not state or 'len' not in state or 'elements' not in state:
        return None
    vector = [0.0 * n for n in range(state['len'])]
    for ind, value in state['elements'].items():
        vector[int(ind)] = value
    return tuple(vector)


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
        for token in text:
            if not token.isalpha():
                text = text.replace(token, " ")
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

        if not documents or not isinstance(documents, list):
            return None
        return [self.tokenize(document) for document in documents if self.tokenize(document)] or None

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """

        if not tokens or not isinstance(tokens, list):
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
        self._vocabulary = sorted(list(set(sum(self._corpus, []))))
        for word in self._vocabulary:
            word_count = sum(1 for document in self._corpus if word in document)
            self._idf_values[word] = math.log((len(self._corpus) - word_count + 0.5) / (word_count + 0.5))
            self._token2ind[word] = self._vocabulary.index(word)
        if not self._vocabulary or not self._idf_values or not self._token2ind:
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
        if not tokenized_document or not isinstance(tokenized_document, list):
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
        result = []
        for i, num in enumerate(vector):
            if num != 0.0:
                for token, ind in self._token2ind.items():
                    if i == ind:
                        result.append(token)
        return result

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """
        if not file_path or not isinstance(file_path, str):
            return False
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({'vocabulary': self._vocabulary, 'idf_values': self._idf_values, 'token2ind': self._token2ind}, file)
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
        if not file_path or not isinstance(file_path, str):
            return False
        with open(file_path, 'r', encoding='utf-8') as file:
            info = json.load(file)
            if 'vocabulary' not in info or 'idf_values' not in info or 'token2ind' not in info:
                return False
            self._vocabulary = info['vocabulary']
            self._idf_values = info['idf_values']
            self._token2ind = info['token2ind']
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
        if not document or not isinstance(document, list):
            return None
        vector_list = [0.0 * n for n in range(len(self._vocabulary))]
        unique_words = list(set(self._vocabulary))
        tf_dict = {token: document.count(token) / len(document) for token in unique_words}
        for ind, word in enumerate(self._vocabulary):
            vector_list[ind] = tf_dict[word] * self._idf_values[word]
        return tuple(vector_list)


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
        if not documents or not isinstance(documents, list):
            return False
        self._documents = documents
        self._document_vectors = [self._index_document(document) for document in documents]
        return False if None in self._document_vectors else True

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
        if (not query or not isinstance(query, str) or not isinstance(n_neighbours, int)
                or isinstance(n_neighbours, bool)):
            return None
        query_vector = self._index_document(query)
        if not query_vector:
            return None
        knn_list = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if not knn_list:
            return None
        result = []
        for tup in knn_list:
            result.append((tup[-1], self._documents[tup[0]]))
        return result

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where to save the instance

        Returns:
            bool: returns True if save was done correctly, False in another cases
        """
        if not file_path or not isinstance(file_path, str):
            return False
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump({'engine': self._dump_documents()}, file)
        return True

    def load(self, file_path: str) -> bool:
        """
        Load engine from state.

        Args:
            file_path (str): The path to the file with state

        Returns:
            bool: True if engine was loaded, False in other cases
        """
        if not file_path or not isinstance(file_path, str):
            return False
        with open(file_path, 'r', encoding='utf-8') as file:
            info = json.load(file)
            if 'engine' not in info:
                return False
            self._documents = info['engine']['documents']
            self._document_vectors = info['engine']['document_vectors']
        return True

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
        document_number = self._calculate_knn(query_vector, self._document_vectors, 1)
        if not document_number:
            return None
        return self._documents[document_number[0][0]]

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
        if (not query_vector or None in query_vector or not document_vectors
                or not isinstance(n_neighbours, int) or isinstance(n_neighbours, bool)):
            return None
        sorted_vectors = [(document_vectors.index(document_vector), calculate_distance(query_vector, document_vector))
                          for document_vector in document_vectors]
        sorted_vectors.sort(key=lambda x: x[-1])
        return [sorted_vectors[ind] for ind in range(0, n_neighbours)]

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
        if not tokenized_doc:
            return None
        return self._vectorizer.vectorize(tokenized_doc)

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        return {'documents': self._documents, 'document_vectors': [save_vector(vector) for vector in self._document_vectors]}

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if not state or not isinstance(state, dict):
            return False
        self._documents = state['documents']
        self._document_vectors = [load_vector(vector) for vector in state['document_vectors']]
        if not self._documents or not self._document_vectors:
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
        if not vectors:
            return False
        depth = 0
        parent = Node()
        is_left = True
        dimensions = len(vectors[0])
        space_condition = [[vectors, depth, parent, is_left, dimensions]]
        while space_condition:
            for ind, space in enumerate(space_condition):
                if not space[0]:
                    continue
                axis = depth % dimensions
                space[0].sort(key=lambda x: x[axis])
                median_index = len(space[0]) // 2
                median_dot = space[0][median_index]
                median_dot_node = Node(median_dot)
                if space[2].payload == -1:
                    self._root = median_dot_node
                else:
                    if space[3]:
                        parent.left_node = median_dot_node
                    else:
                        parent.right_node = median_dot_node
                new_left_space = [space[0][:median_index], depth + 1, median_dot_node, True, dimensions]
                new_right_space = [space[0][median_index + 1:], depth + 1, median_dot_node, False, dimensions]
                space_condition.insert(ind + 1, new_left_space)
                space_condition.insert(ind + 2, new_right_space)
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
        if not vector or not isinstance(k, int) or not isinstance(vector, tuple):
            return None
        nodes = [(self._root, 0)]
        nearest_neighbors = []
        while nodes:
            for node in nodes:
                pair = nodes.pop(0)
                if not pair[0]:
                    return None
                if not pair[0].left_node and not pair[0].right_node:
                    nearest_neighbors.append((calculate_distance(vector, pair[0].vector), pair[0].payload))
                axis = pair[1] % len(pair[0].vector)
                if vector[axis] < pair[0].vector[axis]:
                    if pair[0].left_node is not None:
                        nodes.append((pair[0].left_node, pair[-1] + 1))
                else:
                    if pair[0].right_node is not None:
                        nodes.append((pair[0].right_node, pair[-1] + 1))
        return nearest_neighbors


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
        if not vector or not isinstance(k, int) or not isinstance(vector, tuple):
            return None
        nodes = [(self._root, 0)]
        nearest_neighbors = []
        while nodes:
            for node in nodes:
                pair = nodes.pop(0)
                if not pair[0]:
                    return None
                if not pair[0].left_node and not pair[0].right_node:
                    nearest_neighbors.append((calculate_distance(vector, pair[0].vector), pair[0].payload))
                axis = pair[1] % len(pair[0].vector)
                if vector[axis] < pair[0].vector[axis]:
                    if pair[0].left_node is not None:
                        nodes.append((pair[0].left_node, pair[-1] + 1))
                else:
                    if pair[0].right_node is not None:
                        nodes.append((pair[0].right_node, pair[-1] + 1))
                n = max(sorted([neighbour[0] for neighbour in nearest_neighbors]))
                if (vector[axis] - pair[0].vector[axis]) ** 2 < n:
                    if pair[0].left_node is not None:
                        nodes.append((pair[0].right_node, pair[-1] + 1))
                    else:
                        if pair[0].right_node is not None:
                            nodes.append((pair[0].left_node, pair[-1] + 1))
        while len(nearest_neighbors) > k:
            nearest_neighbors.remove(sorted(nearest_neighbors, reverse=True, key=lambda x: x[0])[0])
        return nearest_neighbors


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
        if not documents or not isinstance(documents, list) or not all(isinstance(text, str) for text in documents):
            return False
        self._documents = documents
        self._document_vectors = [self._index_document(document) for document in documents]
        self._tree.build(self._document_vectors)
        return False if None in self._document_vectors else True

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
        if (not query or not isinstance(query, str) or not isinstance(n_neighbours, int)
                or isinstance(n_neighbours, bool)):
            return None
        query_vector = self._index_document(query)
        if not query_vector or not self._tree.query(query_vector):
            return None
        result = []
        for neighbour in self._tree.query(query_vector):
            result.append((neighbour[0], self._documents[neighbour[1]]))
        return result

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
