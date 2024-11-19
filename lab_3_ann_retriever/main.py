"""
Lab 3.

Vector search with text retrieving
"""
from json import dump, load
from math import sqrt
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
    elements = {}
    for index, value in enumerate(vector):
        if value != 0.0:
            elements[index] = value

    return {
        'len': len(vector),
        'elements': elements
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
    if not isinstance(state, dict) or 'len' not in state or 'elements' not in state:
        return None

    vector = []
    for i in range(state['len']):
        if str(i) not in state['elements']:
            vector.append(0.0)
        else:
            vector.append(state['elements'][str(i)])
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
        if not isinstance(text, str) or not text:
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
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents) \
                or not documents:
            return None

        tokenized_documents: list[list[str]] = []
        for doc in documents:
            tokenized_doc = self.tokenize(doc)
            if tokenized_doc is None:
                return None
            no_stopwords_doc = self._remove_stop_words(tokenized_doc)
            if no_stopwords_doc is None:
                return None
            tokenized_documents.append(no_stopwords_doc)
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
        if not isinstance(tokens, list) or not all(isinstance(item, str) for item in tokens) or \
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

        vocabulary = set()
        for doc in self._corpus:
            vocabulary |= set(doc)
        self._vocabulary = sorted(list(vocabulary))

        self._token2ind = {value: index for index, value in enumerate(self._vocabulary)}

        idf = calculate_idf(self._vocabulary, self._corpus)
        if idf is None:
            return False
        self._idf_values = idf
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
        if not isinstance(tokenized_document, list) or \
                not all(isinstance(item, str) for item in tokenized_document) or \
                not tokenized_document:
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
        if not isinstance(file_path, str) or not file_path:
            return False

        state = {
            'idf_values': self._idf_values,
            'vocabulary': self._vocabulary,
            'token2ind': self._token2ind
        }

        with open(file_path, 'w', encoding='utf-8') as file:
            dump(state, file)
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
        if not isinstance(file_path, str) or not file_path:
            return False

        with open(file_path, 'r', encoding='utf-8') as file:
            state = load(file)

        for attribute in ('vocabulary', 'token2ind', 'idf_values'):
            if attribute not in state:
                return False

        self._vocabulary = state['vocabulary']
        self._token2ind = state['token2ind']
        self._idf_values = state['idf_values']
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
        if not isinstance(document, list) or not all(isinstance(token, str) for token in document) \
                or not document:
            return None

        tf = {}
        for word in set(self._vocabulary) | set(document):
            tf[word] = document.count(word) / len(document)

        return tuple((tf[word] * self._idf_values[word] if word in document else 0.0
                      for word in self._vocabulary))


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
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents) \
                or not documents:
            return False

        self._document_vectors = []
        for doc in documents:
            indexed = self._index_document(doc)
            if indexed is None:
                return False
            self._document_vectors.append(indexed)

        self._documents = documents
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
        if not isinstance(query, str) or not query or not isinstance(n_neighbours, int) or \
                n_neighbours <= 0:
            return None

        query_vectorized = self._index_document(query)
        if query_vectorized is None:
            return None
        self.index_documents(self._documents)
        knn = self._calculate_knn(query_vectorized, self._document_vectors, n_neighbours)
        if knn is None or not knn or not all(isinstance(index, int) or isinstance(distance, float)
                                             for index, distance in knn):
            return None
        relevant_docs = []
        for index, value in knn:
            relevant_docs.append((value, self._documents[index]))
        return relevant_docs

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where to save the instance

        Returns:
            bool: returns True if save was done correctly, False in another cases
        """
        if not isinstance(file_path, str) or not file_path:
            return False

        result = {
            'engine': self._dump_documents()
        }
        with open(file_path, 'w', encoding='utf-8') as file:
            dump(result, file)
        return True

    def load(self, file_path: str) -> bool:
        """
        Load engine from state.

        Args:
            file_path (str): The path to the file with state

        Returns:
            bool: True if engine was loaded, False in other cases
        """
        if not isinstance(file_path, str) or not file_path:
            return False

        with open(file_path, 'r', encoding='utf-8') as file:
            state = load(file)

        return self._load_documents(state)

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
        if not isinstance(document_vectors, list) or not isinstance(n_neighbours, int) or \
                not query_vector or not document_vectors:
            return None

        distances = []
        for index, value in enumerate(document_vectors):
            distance = calculate_distance(query_vector, value)
            if distance is None:
                return None
            distances.append((index, distance))
        distances = sorted(distances, key=lambda tuple_: tuple_[1])
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

        tokenized_document = self._tokenizer.tokenize(document)
        if tokenized_document is None:
            return None
        vectorized_document = self._vectorizer.vectorize(tokenized_document)
        if vectorized_document is None:
            return None
        return vectorized_document

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        return {
            'documents': self._documents,
            'document_vectors': [save_vector(document) for document in self._document_vectors]
        }

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if not isinstance(state, dict) or 'engine' not in state or \
                'documents' not in state['engine'] or 'document_vectors' not in state['engine']:
            return False

        if not state['engine'] or not state['engine']['documents'] or \
                not state['engine']['document_vectors']:
            return False

        self._documents = state['engine']['documents']
        if not self._documents:
            return False
        for vector in state['engine']['document_vectors']:
            loaded_vector = load_vector(vector)
            if not loaded_vector:
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
        self.payload = payload
        self.left_node = left_node
        self.right_node = right_node

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: state of the Node instance
        """
        return {
            'vector': save_vector(self.vector),
            'payload': self.payload,
            'left_node': self.left_node.save() if self.left_node else None,
            'right_node': self.right_node.save() if self.right_node else None
        }

    def load(self, state: dict[str, dict | int]) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict[str, dict | int]): Saved state of the Node

        Returns:
            bool: True if Node was loaded successfully, False in other cases.
        """
        if not isinstance(state, dict) or 'vector' not in state or 'payload' not in state:
            return False
        if not isinstance(state['vector'], dict) or \
                not isinstance(state['payload'], int) or isinstance(state['payload'], bool):
            return False

        loaded_vector = load_vector(state['vector'])
        if loaded_vector is None:
            return False
        self.vector = loaded_vector
        if self.vector is None:
            return False
        self.payload = state['payload']

        left_node = Node()
        right_node = Node()

        # state_left = state['left_node']
        # if state_left:
        #     if left_node.load(state_left):
        #         self.left_node = left_node
        #     else:
        #         return False
        # else:
        #     self.left_node = None
        if state['left_node'] is None:
            self.left_node = None
        elif not isinstance(state['left_node'], dict):
            return False
        else:
            left_node = Node()
            if left_node.load(state['left_node']):
                self.left_node = left_node

        if state['right_node'] is None:
            self.right_node = None
        elif not isinstance(state['right_node'], dict):
            return False
        else:
            right_node = Node()
            if right_node.load(state['right_node']):
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
        if not isinstance(vectors, list) or not vectors:
            return False

        states_info: list[dict] = [{
                           'vectors': [(vector, index) for index, vector in enumerate(vectors)],
                           'depth': 0,
                           'parent': Node(tuple([0.0] * len(vectors[0])), -1),
                           'is_left': True
                       }]

        while states_info:
            current_vectors = states_info[0]['vectors']
            depth = states_info[0]['depth']
            parent = states_info[0]['parent']
            is_left = states_info[0]['is_left']
            states_info.pop(0)
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
        if not isinstance(vector, (list, tuple)) or not isinstance(k, int):
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

        return {
            'root': self._root.save()
        }

    def load(self, state: dict) -> bool:
        """
        Load NaiveKDTree instance from state.

        Args:
            state (dict): saved state of the NaiveKDTree

        Returns:
            bool: True is loaded successfully, False in other cases
        """
        if not isinstance(state, dict) or 'root' not in state:
            return False

        self._root = Node()
        self._root.load(state)
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
        if not isinstance(vector, tuple) or not vector or not isinstance(k, int):
            return None

        subspaces = [(self._root, 0)]
        while subspaces:
            node, depth = subspaces.pop(0)
            if node is None:
                return None
            if node.left_node is None and node.right_node is None:
                distance = calculate_distance(vector, node.vector)
                return [(distance, node.payload)] if distance is not None else None
            axis = depth % len(node.vector)
            new_depth = depth + 1
            if vector[axis] <= node.vector[axis]:
                subspaces.append((node.left_node, new_depth))
            else:
                subspaces.append((node.right_node, new_depth))
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
        if not isinstance(vector, (list, tuple)) or not isinstance(k, int) or not vector:
            return None

        nodes = [(self._root, 0)]
        nearest_nodes = []
        while nodes:
            node, depth = nodes.pop(0)
            if not node:
                continue
            distance = calculate_distance(vector, node.vector)
            if distance is None:
                return None
            if len(nearest_nodes) < k or distance < max(nearest_nodes)[0]:
                nearest_nodes.append((distance, node.payload))
                if len(nearest_nodes) > k:
                    nearest_nodes.sort(key=lambda pair: pair[0])
                    nearest_nodes.pop(-1)
            axis = depth % len(vector)
            new_depth = depth + 1
            if vector[axis] < node.vector[axis]:
                if node.left_node is not None:
                    nodes.append((node.left_node, new_depth))
            elif node.right_node is not None:
                nodes.append((node.right_node, new_depth))
            if (vector[axis] - node.vector[axis]) ** 2 < max(nearest_nodes)[0]:
                if node.left_node is not None:
                    nodes.append((node.right_node, new_depth))
                elif node.right_node is not None:
                    nodes.append((node.left_node, new_depth))
        return nearest_nodes


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
        if not isinstance(documents, list) or not all(isinstance(item, str) for item in documents):
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
        if not isinstance(query, str) or not isinstance(n_neighbours, int):
            return None

        query_indexed = self._index_document(query)
        if query_indexed is None:
            return None
        result = self._tree.query(query_indexed)
        if result is None or not result or not \
                all(isinstance(distance, float) or isinstance(index, int)
                    for distance, index in result):
            return None
        return [(distance, self._documents[document]) for distance, document in result]

    def save(self, file_path: str) -> bool:
        """
        Save the SearchEngine instance to a file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """
        if not isinstance(file_path, str) or not file_path:
            return False

        tree_state = self._tree.save()
        if tree_state is None:
            return False

        documents = super()._dump_documents()['documents']
        document_vectors = super()._dump_documents()['document_vectors']

        state = {
            "engine": {
                "tree": tree_state,
                "documents": documents,
                "document_vectors": document_vectors,
            }
        }

        with open(file_path, 'w', encoding='utf-8') as file:
            dump(state, file)
        return True

    def load(self, file_path: str) -> bool:
        """
        Load a SearchEngine instance from a file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if engine was loaded successfully, False in other cases
        """
        if not isinstance(file_path, str) or not file_path:
            return False

        with open(file_path, 'r', encoding='utf-8') as file:
            engine = load(file)

        if 'engine' not in engine or 'tree' not in engine['engine']:
            return False

        if not self._tree.load(engine['engine']['tree']):
            return False

        return self._load_documents(engine)


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
        BasicSearchEngine.__init__(self, vectorizer, tokenizer)
        self._tree = KDTree()
