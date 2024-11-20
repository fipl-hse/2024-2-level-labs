"""
Lab 3.

Vector search with text retrieving
"""
# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
import json
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
    if not (isinstance(query_vector, tuple) and isinstance(document_vector, tuple)):
        return None
    if not (len(query_vector) > 0 and len(query_vector) == len(document_vector)):
        return 0.0
    size = len(document_vector)
    return sqrt(sum((query_vector[ind] - document_vector[ind]) ** 2 for ind in range(size)))


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save

    Returns:
        dict: A state of the vector to save
    """
    no_null_vec = {ind: value for ind, value in enumerate(vector) if value != 0.0}
    return {"len": len(vector),"elements": no_null_vec}


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if not (isinstance(state, dict) and "len" in state and "elements" in state):
        return None
    filled_vec = [0.0] * state["len"]
    for unique in state["elements"]:
        filled_vec[int(unique)] = state["elements"][unique]
    return tuple(filled_vec)


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
            if symbol.isalpha() or symbol == " " or symbol == "\n":
                continue
            text = text.replace(symbol, " ")
        no_stops_text = self._remove_stop_words(text.lower().split())
        if isinstance(no_stops_text, list):
            return no_stops_text
        return None

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(documents, list) and all(isinstance(doc, str) for doc in documents)):
            return None
        return_list = []
        for doc in documents:
            tokenized_doc = self.tokenize(doc)
            if not isinstance(tokenized_doc, list):
                return None
            return_list.append(tokenized_doc)
        return return_list

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(tokens, list) and len(tokens) > 0):
            return None
        return [token for token in tokens if not token in self._stop_words]


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
        if not isinstance(self._corpus, list):
            return False
        unique_words = set()
        for doc in self._corpus:
            if all(isinstance(token, str) for token in doc):
                unique_words = unique_words | set(doc)
            else:
                return False
        self._vocabulary = sorted(list(unique_words))
        idfs = calculate_idf(self._vocabulary, self._corpus)
        if not isinstance(idfs, dict):
            return False
        self._idf_values = idfs
        self._token2ind = {word:ind for ind, word in enumerate(self._vocabulary)}
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
        if not (isinstance(tokenized_document, list) and len(tokenized_document) > 0):
            return None
        if len(self._vocabulary) == 0 or len(self._idf_values) == 0 or len(self._token2ind) == 0:
            return ()
        vector = self._calculate_tf_idf(tokenized_document)
        if not isinstance(vector, tuple):
            return None
        return vector

    def vector2tokens(self, vector: Vector) -> list[str] | None:
        """
        Recreate a tokenized document based on a vector.

        Args:
            vector (Vector): Vector to decode

        Returns:
            list[str] | None: Tokenized document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(vector, tuple) and len(vector) == len(self._token2ind)):
            return None
        tokens = []
        for token in self._token2ind:
            vector_value = vector[self._token2ind[token]]
            if not isinstance(vector_value, float):
                return None
            if vector_value != 0:
                tokens.append(token)
        return tokens

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """
        if not (isinstance(file_path, str) and len(file_path) > 0 and
                isinstance(self._vocabulary, list) and isinstance(self._idf_values, dict) and
                isinstance(self._token2ind, dict)):
            return False
        with open(file_path, "w", encoding="utf-8") as write_file:
            json.dump({"vocabulary": self._vocabulary, "idf_values": self._idf_values,
                      "token2ind": self._token2ind}, write_file, indent="\t")
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
        if not (isinstance(file_path, str) and len(file_path) > 0):
            return False
        with open(file_path, "r", encoding="utf-8") as read_file:
            objects = json.load(read_file)
        if not ("vocabulary" in objects and "idf_values" in objects and "token2ind" in objects):
            return False
        vocab = objects["vocabulary"]
        idfs = objects["idf_values"]
        token2ind = objects["token2ind"]
        if isinstance(vocab, list) and isinstance(idfs, dict) and isinstance(token2ind, dict):
            self._idf_values = idfs
            self._vocabulary = vocab
            self._token2ind = token2ind
            return True
        return False

    def _calculate_tf_idf(self, document: list[str]) -> Vector | None:
        """
        Get TF-IDF for document.

        Args:
            document (list[str]): Tokenized document to vectorize

        Returns:
            Vector | None: TF-IDF vector for document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(document, list) and isinstance(self._vocabulary, list) and
                isinstance(self._token2ind, dict)):
            return None
        tf = calculate_tf(self._vocabulary, document)
        if not isinstance(tf, dict):
            return None
        vector_to_fill = [0.0] * len(self._vocabulary)
        for word in tf:
            if word in self._token2ind:
                vec_ind = self._token2ind[word]
                if not isinstance(vec_ind, int):
                    return None
                vector_to_fill[vec_ind] = tf[word] * self._idf_values[word]
        return tuple(vector_to_fill)


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
        if not (isinstance(documents, list) and len(documents) > 0):
            return False
        self._documents = documents
        temp_vector_docs = []
        for doc in self._documents:
            vector_doc = self._index_document(doc)
            if not isinstance(vector_doc, tuple):
                return False
            temp_vector_docs.append(vector_doc)
        self._document_vectors = temp_vector_docs
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
        query_vector = self._index_document(query)
        if not isinstance(query_vector, tuple):
            return None
        nearest_docs = self._calculate_knn(query_vector, self._document_vectors, n_neighbours)
        if not (isinstance(nearest_docs, list) and len(nearest_docs) > 0):
            return None
        output_values = []
        for pair in nearest_docs:
            ind, value = pair
            if not (isinstance(ind, int) and isinstance(value, float)):
                return None
            output_values.append((value, self._documents[ind]))
        return output_values

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where to save the instance

        Returns:
            bool: returns True if save was done correctly, False in another cases
        """
        if not (isinstance(file_path, str) and len(file_path) > 0):
            return False
        engine = self._dump_documents()
        if not isinstance(engine, dict):
            return False
        with open(file_path, 'w', encoding='utf-8') as write_file:
            json.dump({"engine": engine}, write_file, indent="\t")
        return True

    def load(self, file_path: str) -> bool:
        """
        Load engine from state.

        Args:
            file_path (str): The path to the file with state

        Returns:
            bool: True if engine was loaded, False in other cases
        """
        if not (isinstance(file_path, str) and len(file_path) > 0):
            return False
        with open(file_path, 'r', encoding='utf-8') as read_file:
            engine = json.load(read_file)
        if not "engine" in engine:
            return False
        return self._load_documents(engine["engine"])

    def retrieve_vectorized(self, query_vector: Vector) -> str | None:
        """
        Retrieve document by vector.

        Args:
            query_vector (Vector): Question vector

        Returns:
            str | None: Answer document

        In case of corrupt input arguments, None is returned.
        """
        if not (isinstance(query_vector, tuple) and
                len(query_vector) == len(self._document_vectors[0])):
            return None
        closest = self._calculate_knn(query_vector, self._document_vectors, 1)
        if not (isinstance(closest, list) and len(closest) > 0):
            return None
        return self._documents[closest[0][0]]

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
        if not (isinstance(query_vector, tuple) and isinstance(document_vectors, list) and
                isinstance(n_neighbours, int) and len(document_vectors) > 0):
            return None
        distances = []
        for ind, vector in enumerate(document_vectors):
            distance = calculate_distance(query_vector, vector)
            if not isinstance(distance, float):
                return None
            distances.append((ind, distance))
        return sorted(distances, key=lambda a: (a[1], a[0]))[:n_neighbours]

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
        token_doc = self._tokenizer.tokenize(document)
        if not isinstance(token_doc, list):
            return None
        doc_vector = self._vectorizer.vectorize(token_doc)
        if not isinstance(doc_vector, tuple):
            return None
        return doc_vector

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states
        """
        no_null_vecs = []
        for vec in self._document_vectors:
            no_null_vecs.append(save_vector(vec))
        return {"documents": self._documents, "document_vectors": no_null_vecs}

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents

        Returns:
            bool: True if documents were loaded, False in other cases
        """
        if not (isinstance(state, dict) and "documents" in state and "document_vectors" in state):
            return False
        self._documents = state["documents"]
        vecs_with_nulls = []
        for no_null_vec in state["document_vectors"]:
            vec = load_vector(no_null_vec)
            if not isinstance(vec, tuple):
                return False
            vecs_with_nulls.append(vec)
        self._document_vectors = vecs_with_nulls
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
        vec_to_save = save_vector(self.vector)
        left_node_saved = self.left_node.save() if not self.left_node is None else None
        right_node_saved = self.right_node.save() if not self.right_node is None else None
        return {"vector": vec_to_save, "payload": self.payload,
                "left_node": left_node_saved, "right_node": right_node_saved}

    def load(self, state: dict[str, dict | int]) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict[str, dict | int]): Saved state of the Node

        Returns:
            bool: True if Node was loaded successfully, False in other cases.
        """
        if not (isinstance(state, dict) and "vector" in state and "payload" in state and
                "left_node" in state and "right_node" in state):
            return False
        normal_vector = load_vector(state["vector"])
        if not isinstance(normal_vector, tuple):
            return False
        self.vector = normal_vector
        self.payload = state["payload"]
        if isinstance(state["left_node"], dict):
            self.left_node = Node()
            self.left_node.load(state["left_node"])
        elif state["left_node"] is None:
            self.left_node = None
        else:
            return False
        if isinstance(state["right_node"], dict):
            self.right_node = Node()
            self.right_node.load(state["right_node"])
        elif state["right_node"] is None:
            self.right_node = None
        else:
            return False
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
        if not (isinstance(vectors, list) and len(vectors) > 0):
            return False
        ind_vecs = []
        for ind, vec in enumerate(vectors):
            ind_vecs.append((ind, vec))
        info_list = [{"vectors": ind_vecs, "depth": 0, "parent_node": Node(), "is_left": True}]
        while len(info_list) > 0:
            current_space = info_list.pop(0)
            if len(current_space["vectors"]) == 0:
                continue
            axis = current_space["depth"] % len(vectors[0])
            current_space["vectors"].sort(key=lambda a: a[1][axis])
            median_index = len(current_space["vectors"]) // 2
            median_ind_vec = current_space["vectors"][median_index]
            node_to_assign = Node(median_ind_vec[1], median_ind_vec[0])
            vectors_value_left = current_space["vectors"][:median_index]
            vectors_value_right = current_space["vectors"][median_index + 1:]
            depth_value = current_space["depth"] + 1
            if current_space["parent_node"].payload == -1:
                self._root = node_to_assign
                parent_value = self._root
            elif current_space["is_left"]:
                current_space["parent_node"].left_node = node_to_assign
                parent_value = current_space["parent_node"].left_node
            else:
                current_space["parent_node"].right_node = node_to_assign
                parent_value = current_space["parent_node"].right_node
            info_list.append({"vectors": vectors_value_left, "depth": depth_value,
                              "parent_node": parent_value, "is_left": True})
            info_list.append({"vectors": vectors_value_right, "depth": depth_value,
                              "parent_node": parent_value, "is_left": False})
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
        if not (isinstance(vector, tuple) and isinstance(k, int)):
            return None
        results = self._find_closest(vector, k)
        if not isinstance(results, list):
            return None
        return results

    def save(self) -> dict | None:
        """
        Save NaiveKDTree instance to state.

        Returns:
            dict | None: state of the NaiveKDTree instance

        In case of corrupt input arguments, None is returned.
        """
        if self._root is None:
            return None
        saved_root = self._root.save()
        if not isinstance(saved_root, dict):
            return None
        return {"root": saved_root}

    def load(self, state: dict) -> bool:
        """
        Load NaiveKDTree instance from state.

        Args:
            state (dict): saved state of the NaiveKDTree

        Returns:
            bool: True is loaded successfully, False in other cases
        """
        if not (isinstance(state, dict) and "root" in state):
            return False
        saved_root = state["root"]
        if not isinstance(saved_root, dict):
            return False
        self._root = Node()
        self._root.load(saved_root)
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
        if not (isinstance(vector, tuple) and isinstance(k, int) and len(vector) > 0 and
                k > 0 and not self._root is None):
            return None
        distance_list = []
        info_list = [{"node": self._root, "depth": 0}]
        while len(info_list) > 0:
            current_node = info_list.pop(0)
            if current_node["node"].left_node is None and current_node["node"].right_node is None:
                distance = calculate_distance(vector, current_node["node"].vector)
                if not isinstance(distance, float):
                    return None
                distance_list.append((distance, current_node["node"].payload))
                break
            axis = current_node["depth"] % len(vector)
            if current_node["node"].right_node is None:
                new_node = current_node["node"].left_node
            elif current_node["node"].left_node is None:
                new_node = current_node["node"].right_node
            elif vector[axis] <= current_node["node"].vector[axis]:
                new_node = current_node["node"].left_node
            else:
                new_node = current_node["node"].right_node
            info_list.append({"node": new_node, "depth": current_node["depth"] + 1})
        return sorted(distance_list)[:k]


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
        if not (isinstance(vector, tuple) and isinstance(k, int) and len(vector) > 0 and k > 0):
            return None
        closest_nodes = []
        info_list = [{"node": self._root, "depth": 0}]
        while len(info_list) > 0:
            current_node = info_list[0]
            distance = calculate_distance(vector, current_node["node"].vector)
            if not isinstance(distance, float):
                return None
            if len(closest_nodes) < k:
                closest_nodes.append((distance, current_node["node"].payload))
            else:
                max_closest = max(closest_nodes, key=lambda a: a[0])
                if distance < max_closest[0]:
                    closest_nodes.remove(max_closest)
                    closest_nodes.append((distance, current_node["node"].payload))
            if current_node["node"].left_node is None and current_node["node"].right_node is None:
                info_list.pop(0)
                continue
            axis = current_node["depth"] % len(vector)
            if current_node["node"].right_node is None:
                near_node = current_node["node"].left_node
                far_node = None
            elif current_node["node"].left_node is None:
                near_node = current_node["node"].right_node
                far_node = None
            elif vector[axis] < current_node["node"].vector[axis]:
                near_node = current_node["node"].left_node
                far_node = current_node["node"].right_node
            else:
                near_node = current_node["node"].right_node
                far_node = current_node["node"].left_node
            info_list.pop(0)
            info_list.append({"node": near_node, "depth": current_node["depth"] + 1})
            new_max_closest = max(closest_nodes, key=lambda a: a[0])
            if (not far_node is None and
                (vector[axis] - current_node["node"].vector[axis]) ** 2 < new_max_closest[0]):
                info_list.append({"node": far_node, "depth": current_node["depth"] + 1})
        return sorted(closest_nodes)


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
        if not isinstance(documents, list):
            return False
        self._documents = documents
        temp_vector_docs = []
        for doc in self._documents:
            vector_doc = self._index_document(doc)
            if not isinstance(vector_doc, tuple):
                return False
            temp_vector_docs.append(vector_doc)
        self._document_vectors = temp_vector_docs
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
        if not (isinstance(query, str) and isinstance(n_neighbours, int) and
                len(query) > 0 and n_neighbours > 0):
            return None
        query_vector = self._index_document(query)
        if not isinstance(query_vector, tuple):
            return None
        search_results = self._tree.query(query_vector, n_neighbours)
        if not (isinstance(search_results, list) and len(search_results) > 0 and
                all(isinstance(pair[0], float) and
                    isinstance(pair[1], int) for pair in search_results)):
            return None
        output_values = []
        for pair in search_results:
            value, ind = pair
            if not (isinstance(ind, int) and isinstance(value, float)):
                return None
            output_values.append((value, self._documents[ind]))
        return output_values

    def save(self, file_path: str) -> bool:
        """
        Save the SearchEngine instance to a file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """
        if not (isinstance(file_path, str) and len(file_path) > 0):
            return False
        tree_state = self._tree.save()
        documents = self._dump_documents()
        if not (isinstance(tree_state, dict) and len(tree_state) > 0):
            return False
        state = {"engine": {"tree": tree_state}}
        state["engine"].update(documents)
        with open(file_path, "w", encoding="utf-8") as write_file:
            json.dump(state, write_file, indent="\t")
        return True

    def load(self, file_path: str) -> bool:
        """
        Load a SearchEngine instance from a file.

        Args:
            file_path (str): The path to the file from which to load the instance

        Returns:
            bool: True if engine was loaded successfully, False in other cases
        """
        if not (isinstance(file_path, str) and len(file_path) > 0):
            return False
        with open(file_path, "r", encoding="utf-8") as read_file:
            state = json.load(read_file)
        if not (isinstance(state, dict) and "engine" in state):
            return False
        eng = state["engine"]
        if not ("documents" in eng and "document_vectors" in eng and "tree" in eng):
            return False
        self._load_documents(
            {"documents": eng["documents"], "document_vectors": eng["document_vectors"]})
        if not (isinstance(eng["tree"], dict) and len(eng["tree"]) > 0):
            return False
        return self._tree.load(eng["tree"])


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
