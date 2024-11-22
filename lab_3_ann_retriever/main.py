"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
import json
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
    distance = 0.0
    for index, value in enumerate(query_vector):
        distance += (value - document_vector[index]) ** 2
    distance = distance**0.5
    if not isinstance(distance, float):
        return None
    return distance


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """
    elements = {index: value for index, value in enumerate(vector) if value != 0.0}
    return {"len": len(vector), "elements": elements}


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if not state or not isinstance(state, dict) or not "len" in state or not "elements" in state:
        return None
    loaded_vector = [0.0 * i for i in range(state["len"])]
    for index, value in state["elements"].items():
        loaded_vector[int(index)] = value
    return tuple(loaded_vector)


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
        if not text or not isinstance(text, str):
            return None
        for letter in text:
            if not letter.isalpha() and letter != " ":
                text = text.replace(letter, " ")
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
        if (
            not documents
            or not isinstance(documents, list)
            or not all(isinstance(document, str) for document in documents)
        ):
            return None
        tokenized_all = []
        for text in documents:
            tokenized_text = self.tokenize(text)
            if tokenized_text is None:
                return None
            tokenized_all.append(tokenized_text)
        return tokenized_all

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
        return [word for word in tokens if not word in self._stop_words]


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
        self._vocabulary = sorted(list({word for text in self._corpus for word in text}))
        idf = calculate_idf(self._vocabulary, self._corpus)
        if idf is None:
            return False
        self._idf_values = idf
        self._token2ind = {term: index for index, term in enumerate(self._vocabulary)}
        if None in (self._vocabulary, self._idf_values, self._token2ind):
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
        if not vector or len(self._token2ind) != len(vector):
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
        if not file_path or not isinstance(file_path, str):
            return False
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "idf_values": self._idf_values,
                    "vocabulary": self._vocabulary,
                    "token2ind": self._token2ind,
                },
                file,
            )
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
        with open(file_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
            if (
                not "vocabulary" in loaded
                or not "idf_values" in loaded
                or not "token2ind" in loaded
            ):
                return False
            self._idf_values = loaded["idf_values"]
            self._vocabulary = loaded["vocabulary"]
            self._token2ind = loaded["token2ind"]
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
        tf_dict = {word: document.count(word) / len(document) for word in self._vocabulary}
        calculated_list = []
        for word in self._vocabulary:
            calculated_list.append(self._idf_values[word] * tf_dict[word])
        return tuple(calculated_list)


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
        if (
            not documents
            or not isinstance(documents, list)
            or not all(isinstance(word, str) for word in documents)
        ):
            return False
        self._documents = documents
        self._document_vectors = [vec for doc in documents if (vec := self._index_document(doc))]
        if not self._documents or not self._document_vectors:
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
        if not query or not isinstance(query, str) or not isinstance(n_neighbours, int):
            return None
        query_vector = self._index_document(query)
        if query_vector is None:
            return None
        knn = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if not knn or any(pair[1] is None for pair in knn):
            return None
        relevant_documents = [(pair[1], self._documents[pair[0]]) for pair in knn]
        return relevant_documents

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
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump({"engine": self._dump_documents()}, file)
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
        with open(file_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
            if not self._load_documents(loaded):
                return False
            self._load_documents(loaded)
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
            if len(query_vector) != len(vector):
                return None
        doc_num = self._calculate_knn(query_vector, self._document_vectors, 1)
        if not doc_num:
            return None
        return self._documents[doc_num[0][0]]

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
        if not query_vector or not isinstance(query_vector, tuple):
            return None
        if not document_vectors or not isinstance(document_vectors, list):
            return None
        if not isinstance(n_neighbours, int):
            return None
        knn_list = [
            (document_vectors.index(vector), distance)
            for vector in document_vectors
            if (distance := calculate_distance(query_vector, vector)) is not None
        ]
        knn_list.sort(key=lambda x: x[-1])
        if not knn_list:
            return None
        return knn_list[:n_neighbours]

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
        tokenized = self._tokenizer.tokenize(document)
        if not tokenized:
            return None
        return self._vectorizer.vectorize(tokenized)

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        document_vectors = [save_vector(vector) for vector in self._document_vectors]
        return {"documents": self._documents, "document_vectors": document_vectors}

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if not state or not isinstance(state, dict) or "engine" not in state:
            return False
        if "documents" not in state["engine"] or "document_vectors" not in state["engine"]:
            return False
        if not state["engine"]["documents"] or not state["engine"]["document_vectors"]:
            return False
        self._documents = state["engine"]["documents"]
        self._document_vectors = [
            vector
            for document_vector in state["engine"]["document_vectors"]
            if (vector := load_vector(document_vector))
        ]
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
        if self.left_node:
            saved_left = self.left_node.save()
        else:
            saved_left = None
        if self.right_node:
            saved_right = self.right_node.save()
        else:
            saved_right = None
        return {
            "vector": save_vector(self.vector),
            "payload": self.payload,
            "left_node": saved_left,
            "right_node": saved_right,
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
        if (
            "vector" not in state
            or "payload" not in state
            or "left_node" not in state
            or "right_node" not in state
            or not isinstance(state["vector"], dict)
        ):
            return False
        loaded_vector = load_vector(state["vector"])
        if not loaded_vector:
            return False
        self.vector = loaded_vector
        payload = state["payload"]
        if payload and isinstance(payload, int):
            self.payload = payload
        left = Node()
        right = Node()
        if state["left_node"]:
            if not isinstance(state["left_node"], dict) or not left.load(state["left_node"]):
                return False
            self.left_node = left
        else:
            self.left_node = None
        if state["right_node"]:
            if not isinstance(state["right_node"], dict) or not right.load(state["right_node"]):
                return False
            self.right_node = right
        else:
            self.right_node = None
        return bool(self.vector and self.payload)


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
        if (
            not isinstance(vectors, list)
            or not vectors
            or not all(isinstance(tuples, tuple) for tuples in vectors)
        ):
            return False
        if len(vectors) == 0:
            return False
        vectors_remade = []
        for index, vector in enumerate(vectors):
            vectors_remade.append((index, vector))
        start = [(vectors_remade, 0, Node(), True)]
        while start:
            used_info = start.pop(0)
            if not isinstance(used_info, tuple):
                return False
            if not used_info[0]:
                continue
            axis = used_info[1] % len(vectors[0])
            used_info[0].sort(key=lambda a: a[1][axis])
            median_index = len(used_info[0]) // 2
            median_index_vector = used_info[0][median_index]
            new_node = Node(median_index_vector[1], median_index_vector[0])

            if used_info[2].payload == -1:
                self._root = new_node
                for_parent = self._root
            else:
                if used_info[3]:
                    used_info[2].left_node = new_node
                    for_parent = used_info[2].left_node
                else:
                    used_info[2].right_node = new_node
                    for_parent = used_info[2].right_node
            start.append((used_info[0][:median_index], used_info[1] + 1, for_parent, True))
            start.append((used_info[0][median_index + 1 :], used_info[1] + 1, for_parent, False))
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
        if not isinstance(vector, tuple) or not isinstance(k, int) or not vector or not k:
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
        if not state or not isinstance(state, dict) or not "root" in state:
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
        if not isinstance(vector, tuple) or not isinstance(k, int) or not vector:
            return None
        subspace = [(self._root, 0)]
        dimension = len(vector)
        closest = []
        while subspace:
            popped_node = subspace.pop(0)
            used_node, depth = popped_node
            if not isinstance(used_node, Node):
                return None
            if not used_node.left_node and not used_node.right_node:
                distance = calculate_distance(vector, used_node.vector)
                if not isinstance(distance, float):
                    return None
                closest.append((distance, used_node.payload))
            else:
                axis = depth % dimension
                if vector[axis] <= used_node.vector[axis]:
                    if used_node.left_node is not None:
                        subspace.append((used_node.left_node, depth + 1))
                else:
                    if used_node.right_node is not None:
                        subspace.append((used_node.right_node, depth + 1))
        return closest


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
        if (
            not isinstance(vector, tuple)
            or not isinstance(k, int)
            or not vector
            or not k
            or len(vector) == 0
        ):
            return None
        neighbours = []
        dimensions = len(vector)
        start = [(self._root, 0)]
        while start:
            node = start[0][0]
            depth = start[0][1]
            start.pop(0)
            if not node:
                continue
            distance = calculate_distance(vector, node.vector)
            if not distance:
                return None
            if len(neighbours) < k or distance < max(neighbours, key=lambda x: float(x[0]))[0]:
                neighbours.append((distance, node.payload))
                if len(neighbours) > k:
                    neighbours.remove(sorted(neighbours, reverse=False, key=lambda x: x[0])[-1])
            axis = depth % dimensions
            if vector[axis] < node.vector[axis]:
                if node.left_node:
                    start.append((node.left_node, depth + 1))
            else:
                if node.right_node:
                    start.append((node.right_node, depth + 1))
            if (vector[axis] - node.vector[axis]) ** 2 < max(neighbours, key=lambda x: x[0])[0]:
                if node.left_node:
                    start.append((node.right_node, depth + 1))
                else:
                    if node.right_node:
                        start.append((node.right_node, depth + 1))
        return neighbours


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
        if (
            not isinstance(documents, list)
            or not all(isinstance(text, str) for text in documents)
            or not documents
        ):
            return False
        self._documents = documents
        self._document_vectors = [
            vector
            for document in documents
            if (vector := self._index_document(document)) is not None
        ]
        self._tree.build(self._document_vectors)
        if not self._documents or not self._document_vectors:
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
        if not isinstance(query, str) or not isinstance(n_neighbours, int):
            return None

        query_vector = self._index_document(query)
        if query_vector is None:
            return None
        answer = self._tree.query(query_vector, n_neighbours)
        if answer is None:
            return None

        relevant_documents = []
        for _, (distance, index) in enumerate(answer):
            if index is not None and distance is not None:
                relevant_documents.append((distance, self._documents[index]))
        if not relevant_documents:
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
        if not file_path or not isinstance(file_path, str):
            return False
        documents = self._dump_documents()["documents"]
        document_vectors = self._dump_documents()["document_vectors"]
        if not self._tree.save():
            return False
        state = {
            "engine": {
                "tree": self._tree.save(),
                "documents": documents,
                "document_vectors": document_vectors,
            }
        }
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(state, file)
        return True

    def load(self, file_path: str) -> bool:
        """
        Load a SearchEngine instance from a file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if engine was loaded successfully, False in other cases
        """
        if not file_path or not isinstance(file_path, str):
            return False
        with open(file_path, "r", encoding="utf-8") as file:
            loaded = json.load(file)
            if (
                "engine" not in loaded
                or "tree" not in loaded["engine"]
                or "documents" not in loaded["engine"]
                or "document_vectors" not in loaded["engine"]
            ):
                return False
        self._load_documents(loaded)
        if not self._tree.load(loaded["engine"]["tree"]):
            return False
        self._tree.load(loaded["engine"]["tree"])
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
