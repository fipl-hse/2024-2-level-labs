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
    if (query_vector is None or document_vector is None
            or not all(isinstance(elem, float) for elem in query_vector)
            or not all(isinstance(elem, float) for elem in document_vector)):
        return None
    if not query_vector or not document_vector:
        return 0.0
    distance = math.sqrt(sum((query_vector[index] - document_vector[index])**2
                         for index in range(len(document_vector))))
    return distance


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """
    index_value = {}
    for index, value in enumerate(vector):
        if value == 0.0:
            continue
        index_value[index] = value
    result = {"len": len(vector),
              "elements": index_value}
    return result


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if not isinstance(state, dict):
        return None
    elements = state.get("elements")
    len_vector = state.get("len")
    if not isinstance(elements, dict) or not isinstance(len_vector, int):
        return None
    vector = [0.0 * x for x in range(len_vector)]
    for index, value in elements.items():
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
        text = text.lower()
        for elem in text:
            if not elem.isalpha() and elem != ' ':
                text = text.replace(elem, ' ')
        clear_tokens = self._remove_stop_words(text.lower().split())
        return clear_tokens

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
                or not all(isinstance(doc, str) for doc in documents)):
            return None
        result = []
        for doc in documents:
            doc_tokens = self.tokenize(doc)
            if not isinstance(doc_tokens, list) or doc_tokens is None:
                return None
            result.append(doc_tokens)
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
        if (not isinstance(tokens, list)
                or not all(isinstance(token, str) for token in tokens)
                or len(tokens) == 0):
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
        self.build()

    def build(self) -> bool:
        """
        Build vocabulary with tokenized_documents.

        Returns:
            bool: True if built successfully, False in other case
        """
        if (not isinstance(self._corpus, list) or not self._corpus
                or not all(isinstance(tokens, list) for tokens in self._corpus)
                or not all(isinstance(word, str) for tokens in self._corpus for word in tokens)):
            return False
        self._vocabulary = sorted(list(set(sum(self._corpus, []))))
        for tokens in self._corpus:
            for word in tokens:
                if word in self._vocabulary:
                    continue
                self._vocabulary.append(word)
        self._vocabulary.sort()
        if not isinstance(self._vocabulary, list):
            return False
        idf = calculate_idf(self._vocabulary, self._corpus)
        if idf is None or not isinstance(idf, dict):
            return False
        self._idf_values = idf
        for token in self._vocabulary:
            self._token2ind[token] = self._vocabulary.index(token)
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
        if not isinstance(tokenized_document, list) or len(tokenized_document) == 0\
                or not all(isinstance(word, str) for word in tokenized_document):
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
        for word, index in self._token2ind.items():
            if not isinstance(index, int):
                return None
            if vector[index] != 0.0:
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
        if not isinstance(file_path, str):
            return False
        with open(file_path, 'w', encoding='utf-8') as document:
            json.dump({"vocabulary": self._vocabulary,
                       "idf_values": self._idf_values,
                       "token2ind": self._token2ind}, document)
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
        with open(file_path, "r", encoding="utf-8") as document:
            data = json.load(document)
            if 'vocabulary' not in data or 'idf_values' not in data or 'token2ind' not in data:
                return False
            self._vocabulary = data["vocabulary"]
            self._idf_values = data["idf_values"]
            self._token2ind = data["token2ind"]
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
        if (not isinstance(document, list) or len(document) == 0
                or not all(isinstance(word, str) for word in document)):
            return None
        tf = calculate_tf(self._vocabulary, document)
        if not isinstance(tf, dict):
            return None
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
        self._tokenizer = tokenizer
        self._vectorizer = vectorizer
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
        if (not isinstance(documents, list) or not documents
                or not all(isinstance(text, str) for text in documents)):
            return False
        self._documents = documents
        for text in documents:
            vector = self._index_document(text)
            if not isinstance(vector, tuple):
                return False
            self._document_vectors.append(vector)
        if not self._document_vectors or self._document_vectors is None:
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
        if (not isinstance(query, str) or not query
                or not isinstance(n_neighbours, int) or n_neighbours <= 0):
            return None
        query_vector = self._index_document(query)
        if query_vector is None or not query_vector:
            return None
        distances = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if distances is None or not distances:
            return None
        result = []
        for distance in distances:
            if not isinstance(distance, tuple) or not isinstance(distance[0], int):
                return None
            result.append((distance[1], self._documents[distance[0]]))
        return result

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
        with open(file_path, "w", encoding="utf-8") as document:
            json.dump({"engine": self._dump_documents()}, document)
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
        with open(file_path, 'r', encoding='utf-8') as document:
            data = json.load(document)
            self._load_documents(data)
            if not self._load_documents(data):
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
        if not query_vector or not isinstance(query_vector, tuple)\
                or len(query_vector) != len(self._document_vectors[0]):
            return None
        document = self._calculate_knn(query_vector, self._document_vectors, 1)
        if not document or document is None:
            return None
        pair = document[0]
        return self._documents[pair[0]]

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
        if not isinstance(document_vectors, list) or not isinstance(n_neighbours, int):
            return None
        if not query_vector or query_vector is None \
                or not document_vectors or document_vectors is None:
            return None
        if not all(isinstance(num, float) for num in query_vector)\
                or not all(isinstance(num, float) for vector in document_vectors
                           for num in vector):
            return None
        distances = [calculate_distance(query_vector, vector) for vector in document_vectors]
        distances_with_indexes = []
        for distance in distances:
            if not isinstance(distance, float):
                return None
            distances_with_indexes.append((distances.index(distance), distance))
        result = sorted(distances_with_indexes, key=lambda x: x[1])
        return result[:n_neighbours]

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
        tokens = self._tokenizer.tokenize(document)
        if not tokens:
            return None
        return self._vectorizer.vectorize(tokens)

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        return {"documents": self._documents,
                "document_vectors": [save_vector(vector) for vector in self._document_vectors]}

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if (not isinstance(state, dict) or "engine" not in state
                or "documents" not in state["engine"] or "document_vectors" not in state["engine"]):
            return False
        if not state['engine']['documents'] or not state['engine']['document_vectors']:
            return False
        documents = state["engine"]["documents"]
        if not documents:
            return False
        self._documents = documents
        document_vectors = [load_vector(vector) for vector in state['engine']['document_vectors']]
        if not document_vectors:
            return False
        self._document_vectors = document_vectors
        if not self._documents or not self._document_vectors or None in self._document_vectors:
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
        self.left_node = left_node
        self.right_node = right_node
        self.payload = payload

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: state of the Node instance
        """
        return {"vector": save_vector(self.vector),
                "payload": self.payload,
                "left_node": self.left_node.save() if self.left_node else None,
                "right_node": self.right_node.save() if self.right_node else None}

    def load(self, state: dict[str, dict | int]) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict[str, dict | int]): Saved state of the Node

        Returns:
            bool: True if Node was loaded successfully, False in other cases.
        """
        if not isinstance(state, dict):
            return False
        if "vector" not in state or "payload" not in state\
                or "left_node" not in state or "right_node" not in state:
            return False
        vector = load_vector(state["vector"])
        if not isinstance(vector, tuple):
            return False
        self.vector = vector
        payload = state["payload"]
        if not isinstance(payload, int):
            return False
        self.payload = payload
        left_node = Node()
        right_node = Node()
        if state["left_node"] is not None:
            if not left_node.load(state["left_node"]):
                return False
            self.left_node = left_node
        else:
            self.left_node = None
        if state["right_node"] is not None:
            if not right_node.load(state["right_node"]):
                return False
            self.right_node = right_node
        else:
            self.right_node = None
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
        if not vectors:
            return False
        dimensions = len(vectors[0])
        spaces = [(vectors, 0, Node(), True)]
        while spaces:
            space = spaces.pop()
            space_vectors = space[0]
            current_parent = space[2]
            current_depth = space[1]
            if (not isinstance(space_vectors, list) or not space_vectors
                    or not current_parent or not isinstance(current_depth, int)
                    or not isinstance(dimensions, int)):
                continue
            axis = current_depth % dimensions
            space_vectors_sorted = sorted(space_vectors, key=lambda vector: vector[axis])
            median_index = len(space_vectors_sorted) // 2
            median_dot = space_vectors_sorted[median_index]
            node_median_dot = Node(median_dot, vectors.index(median_dot))
            if current_parent.payload == -1:
                self._root = node_median_dot
            else:
                if space[3]:
                    current_parent.left_node = node_median_dot
                else:
                    current_parent.right_node = node_median_dot
            left_space = (space_vectors_sorted[:median_index], (current_depth + 1),
                          node_median_dot, True)
            right_space = (space_vectors_sorted[median_index + 1:], (current_depth + 1),
                           node_median_dot, False)
            spaces.append(left_space)
            spaces.append(right_space)
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
        if not self._root:
            return None
        return {"root": self._root.save()}

    def load(self, state: dict) -> bool:
        """
        Load NaiveKDTree instance from state.

        Args:
            state (dict): saved state of the NaiveKDTree

        Returns:
            bool: True is loaded successfully, False in other cases
        """
        if not isinstance(state, dict) or "root" not in state:
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
        if not vector or not isinstance(k, int) or not isinstance(vector, tuple):
            return None
        space = [(self._root, 0)]
        result = []
        while space:
            node, depth = space.pop()
            if node is None or not isinstance(node.payload, int):
                return None
            if not node.left_node and not node.right_node:
                distance = calculate_distance(vector, node.vector)
                if not isinstance(distance, float):
                    return None
                result.append((distance, node.payload))
                if len(result) == k:
                    return result
            axis = depth % len(node.vector)
            if vector[axis] <= node.vector[axis]:
                if node.left_node is not None:
                    space.append((node.left_node, depth + 1))
            else:
                if node.right_node is not None:
                    space.append((node.right_node, depth + 1))
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
        if not isinstance(vector, tuple) or not isinstance(k, int) or not vector:
            return None
        tree = [(self._root, 0)]
        best = []
        while tree:
            node, depth = tree.pop()
            if not node:
                continue
            distance = calculate_distance(vector, node.vector)
            if not isinstance(distance, float):
                return None
            if not best:
                best.append((distance, node.payload))
            elif distance < max(best, key=lambda x: x[0])[0] and len(best) < k:
                best.append((distance, node.payload))
            elif distance < max(best, key=lambda x: x[0])[0] and len(best) >= k:
                best.sort(key=lambda x: x[0])
                best.pop()
                best.append((distance, node.payload))
            axis = depth % len(node.vector)
            if vector[axis] <= node.vector[axis]:
                tree.append((node.left_node, depth + 1))
                if ((vector[axis] - node.vector[axis]) ** 2) < max([pair[0] for pair in best]):
                    tree.append((node.right_node, depth + 1))
            else:
                tree.append((node.right_node, depth + 1))
                if ((vector[axis] - node.vector[axis]) ** 2) < max([pair[0] for pair in best]):
                    tree.append((node.left_node, depth + 1))
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
        self._tree = KDTree()
        BasicSearchEngine.__init__(self, vectorizer, tokenizer)

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for retriever.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if document is successfully indexed

        In case of corrupt input arguments, False is returned.
        """
        if (not isinstance(documents, list)
                or not all(isinstance(text, str) for text in documents)):
            return False
        self._documents = documents
        self._document_vectors = []
        for text in documents:
            vector = self._index_document(text)
            if not vector:
                return False
            self._document_vectors.append(vector)
        if not self._document_vectors or self._document_vectors is None\
                or None in self._document_vectors:
            return False
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
        if not isinstance(query, str) or not isinstance(n_neighbours, int) or query is None:
            return None
        result = []
        query_vector = self._index_document(query)
        if query_vector is None:
            return None
        neighbours = self._tree.query(query_vector, n_neighbours)
        if not neighbours or None in neighbours:
            return None
        for distance, index in neighbours:
            if not isinstance(distance, float) or not isinstance(index, int):
                return None
            result.append((distance, self._documents[index]))
        return result

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
        if not self._tree.save():
            return False
        with open(file_path, "w", encoding="utf-8") as document:
            json.dump({"engine": {"tree": self._tree.save(),
                                      "documents": self._dump_documents()["documents"],
                                      "document_vectors": self._dump_documents()["document_vectors"]}},
                      document)
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
        with open(file_path, "r", encoding="utf-8") as document:
            data = json.load(document)
            if not isinstance(data, dict) or "engine" not in data or "tree" not in data["engine"]:
                return False
            if not self._tree.load(data['engine']['tree']):
                return False
            return self._load_documents(data)


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
