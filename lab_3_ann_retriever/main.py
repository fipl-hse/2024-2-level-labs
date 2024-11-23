"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
import json
import math
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
    return math.sqrt(sum((value - document_vector[i]) ** 2 for i, value in enumerate(query_vector)))


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """
    return {
        'len': len(vector),
        'elements': {i: value for i, value in enumerate(vector) if value}
    }


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if not state or not isinstance(state, dict) or 'len' not in state or 'elements' not in state:
        return None
    vector = [0.0] * state['len']
    for index, value in state['elements'].items():
        vector[int(index)] = value
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
        for symbol in text:
            if not symbol.isalpha() and symbol != ' ':
                text = text.replace(symbol, ' ')
        without_sp = self._remove_stop_words(text.lower().split())
        if without_sp is None:
            return None
        return without_sp

        if not isinstance(text, str):
            return None
        tokens = []
        for symbol in text:
            if symbol.isalpha() or symbol == ' ':
                tokens.append(symbol.lower())
        text_tokens = ''.join(tokens).split()
        return self._remove_stop_words(text_tokens)

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

        if not isinstance(documents, list) or not documents:
            return None
        if not all(isinstance(doc, str) for doc in documents):
            return None
        tokenized_documents = [self.tokenize(doc) for doc in documents]
        if any(doc is None for doc in tokenized_documents):
            return None

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
        if not vector or not isinstance(vector, tuple) \
                or not all(isinstance(elem, float) for elem in vector) \
                or len(vector) != len(self._token2ind):
            return None
        return [token for i, value in enumerate(vector) if value for token, index
                in self._token2ind.items() if index == i]

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
        state = {
            'idf_values': self._idf_values,
            'vocabulary': self._vocabulary,
            'token2ind': self._token2ind
        }
        with open(file_path, 'w', encoding='utf-8') as file_to_save:
            json.dump(state, file_to_save)
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
        with open(file_path, 'r', encoding='utf-8') as file_to_read:
            state = json.load(file_to_read)
        if not state or not isinstance(state, dict) or 'idf_values' not in state or \
                'vocabulary' not in state or 'token2ind' not in state:
            return False
        self._idf_values = state['idf_values']
        self._vocabulary = state['vocabulary']
        self._token2ind = state['token2ind']
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
            vector = self._index_document(doc)
            if vector is None or None in vector:
                return False
            vectors.append(vector)
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
        if not isinstance(file_path, str):
            return False
        engine_state = {'engine': self._dump_documents()}
        with open(file_path, 'w', encoding='utf-8') as file_to_save:
            json.dump(engine_state, file_to_save)
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
        with open(file_path, 'r', encoding='utf-8') as file_to_read:
            state = json.load(file_to_read)
        if not state or not isinstance(state, dict) or 'engine' not in state or \
                'documents' not in state['engine'] or \
                'document_vectors' not in state['engine']:
            return False
        self._load_documents(state)
        if not self._document_vectors or not self._documents:
            return False
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
        return self._vectorizer.vectorize(tokenized_doc)

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        return {
            'documents': self._documents,
            'document_vectors': [save_vector(vector) for vector in self._document_vectors]
        }

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if not state or not isinstance(state, dict) or 'engine' not in state or \
                'documents' not in state['engine'] or \
                'document_vectors' not in state['engine']:
            return False
        self._documents = state['engine']['documents']
        for vector in state['engine']['document_vectors']:
            loaded_vector = load_vector(vector)
            if loaded_vector is None or not loaded_vector:
                return False
            self._document_vectors.append(loaded_vector)
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
        self.left_node = left_node
        self.right_node = right_node
        self.payload = payload

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: state of the Node instance
        """
        return {
            'vector': save_vector(self.vector),
            'payload': self.payload,
            'left_node': self.left_node.save() if self.left_node is not None else None,
            'right_node': self.right_node.save() if self.right_node is not None else None
        }

    def load(self, state: dict[str, dict | int]) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict[str, dict | int]): Saved state of the Node

        Returns:
            bool: True if Node was loaded successfully, False in other cases.
        """
        if not state or not isinstance(state, dict):
            return False
        if 'payload' not in state or 'vector' not in state \
                or not state['vector'] or not isinstance(state['vector'], dict):
            return False
        vector = load_vector(state['vector'])
        payload = state['payload']
        if not vector or not isinstance(vector, tuple) \
                or not all(isinstance(value, float) for value in vector) \
                or not isinstance(payload, int):
            return False
        self.vector = vector
        self.payload = payload
        if state['left_node'] is None:
            self.left_node = None
        else:
            if not isinstance(state['left_node'], dict):
                return False
            left_node = Node()
            left_node.load(state['left_node'])
            self.left_node = left_node
        if state['right_node'] is None:
            self.right_node = None
        else:
            if not isinstance(state['right_node'], dict):
                return False
            right_node = Node()
            right_node.load(state['right_node'])
            self.right_node = right_node
        return True


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
        dim = len(vectors[0])
        all_dimensions = [(vectors, 0, Node(), True)]
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
            all_dimensions.append((sorted_vectors[:median_index], dimension[1] + 1, node_median,
                                   True))
            all_dimensions.append((sorted_vectors[median_index + 1:], dimension[1] + 1, node_median,
                                   False))
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
        return self._find_closest(vector, k)

    def save(self) -> dict | None:
        """
        Save NaiveKDTree instance to state.

        Returns:
            dict | None: state of the NaiveKDTree instance

        In case of corrupt input arguments, None is returned.
        """
        if self._root is None:
            return None
        root = self._root.save()
        if not root or root is None or not isinstance(root, dict):
            return None
        return {'root': root}

    def load(self, state: dict) -> bool:
        """
        Load NaiveKDTree instance from state.

        Args:
            state (dict): saved state of the NaiveKDTree

        Returns:
            bool: True is loaded successfully, False in other cases
        """
        if not state or not isinstance(state, dict) or 'root' not in state:
            return False
        self._root = Node()
        self._root.load(state['root'])
        if not self._root:
            return False
        return True

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
                if len(result) == k:
                    return result
            axis = depth % len(vector)
            if vector[axis] <= node.vector[axis]:
                if node.left_node is not None:
                    nodes.append((node.left_node, depth + 1))
            else:
                if node.right_node is not None:
                    nodes.append((node.right_node, depth + 1))
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
        if not vector or not isinstance(vector, tuple) or not isinstance(k, int):
            return None
        current_pairs = [(self._root, 0)]
        closest_nodes = []
        while current_pairs:
            node, depth = current_pairs.pop()
            if node is None or not isinstance(node.payload, int):
                continue
            distance = calculate_distance(vector, node.vector)
            if not isinstance(distance, float):
                return None
            if not closest_nodes:
                closest_nodes.append((distance, node.payload))
            if len(closest_nodes) < k and distance < max(closest_nodes, key=lambda x: x[0])[0]:
                closest_nodes.append((distance, node.payload))
            if len(closest_nodes) >= k and distance < max(closest_nodes, key=lambda x: x[0])[0]:
                closest_nodes.sort(reverse=True, key=lambda x: x[0])
                closest_nodes.pop(0)
                closest_nodes.append((distance, node.payload))
            axis = depth % len(vector)
            if vector[axis] <= node.vector[axis]:
                current_pairs.append((node.left_node, depth + 1))
                if ((vector[axis] - node.vector[axis]) ** 2) < max(closest_nodes,
                                                                   key=lambda x: x[0])[0]:
                    current_pairs.append((node.right_node, depth + 1))
            else:
                current_pairs.append((node.right_node, depth + 1))
                if ((vector[axis] - node.vector[axis]) ** 2) < max(closest_nodes,
                                                                   key=lambda x: x[0])[0]:
                    current_pairs.append((node.left_node, depth + 1))
        return closest_nodes


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
        query_vector = self._index_document(query)
        if query_vector is None:
            return None
        most_relevant = self._tree.query(query_vector, n_neighbours)
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
        if not isinstance(file_path, str):
            return False
        tree_state = self._tree.save()
        if not tree_state or tree_state is None:
            return False
        dump = self._dump_documents()
        documents = dump['documents']
        document_vectors = dump['document_vectors']
        if not documents or documents is None or not document_vectors or document_vectors is None:
            return False
        state = {
            "engine": {
                "tree": tree_state,
                "documents": documents,
                "document_vectors": document_vectors,
            }
        }
        with open(file_path, 'w', encoding='utf-8') as file_to_save:
            json.dump(state, file_to_save)
        return True

    def load(self, file_path: str) -> bool:
        """
        Load a SearchEngine instance from a file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if engine was loaded successfully, False in other cases
        """
        if not isinstance(file_path, str):
            return False
        with open(file_path, 'r', encoding='utf-8') as file_to_read:
            state = json.load(file_to_read)
        if not isinstance(state, dict) or 'engine' not in state or 'tree' not in state['engine']:
            return False
        if not self._tree.load(state['engine']['tree']) or not self._load_documents(state):
            return False
        return True


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
        SearchEngine.__init__(self, vectorizer, tokenizer)
        self._tree = KDTree()
