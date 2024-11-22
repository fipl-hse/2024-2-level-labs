"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from math import sqrt
from typing import Protocol

from lab_2_retrieval_w_bm25.main import calculate_idf

Vector = tuple[float, ...]


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

    return sqrt(sum((query_val - doc_val) ** 2 for query_val, doc_val
                    in zip(query_vector, document_vector)))


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
        if (not isinstance(stop_words, list)
                or not all(isinstance(item, str) for item in stop_words) or not stop_words):
            raise ValueError
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

        for element in text:
            if not element.isalpha() and not element.isspace():
                text = text.replace(element, ' ')
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
        if (not isinstance(documents, list)
                or not all(isinstance(item, str)
                           for item in documents) or not documents):
            return None
        doc = []
        for text in documents:
            processed_text = self.tokenize(text)
            if processed_text is None:
                return None
            doc.append(processed_text)
        return doc

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if (not isinstance(tokens, list) or not
        all(isinstance(item, str) for item in tokens) or not tokens):
            return None
        filtered_tokens = [word for word in tokens if word not in set(self._stop_words)]
        return filtered_tokens


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

        voc = set()
        for doc in self._corpus:
            if all(isinstance(elem, str) for elem in doc):
                voc.update(doc)
            else:
                return False

        self._vocabulary = sorted(list(voc))
        self._idf_values = calculate_idf(self._vocabulary, self._corpus)
        if self._idf_values is None:
            return False

        self._token2ind = {value: ind for ind, value in enumerate(self._vocabulary)}

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
        if (not isinstance(tokenized_document, list) or
                not all(isinstance(item, str) for item in tokenized_document)
                or not tokenized_document):
            return None

        vect_doc = self._calculate_tf_idf(tokenized_document)
        if vect_doc is None:
            return None
        return vect_doc

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

        toks = []

        for item in self._vocabulary:
            if vector[self._token2ind[item]] != 0.0:
                toks.append(item)
        return toks

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
        if (not isinstance(document, list) or not
        all(isinstance(item, str) for item in document) or not document):
            return None

        tf_idf_vector = []

        for item in self._vocabulary:
            tf = document.count(item) / len(document)
            idf = 0.0
            if item in self._idf_values:
                idf = self._idf_values[item]
            tf_idf_vector.append(tf * idf)

        return tuple(tf_idf_vector)


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
        if (not isinstance(query, str) or
                not isinstance(n_neighbours, int)
                or not query or n_neighbours <= 0):
            return None

        quer_vec = self._index_document(query)
        if not quer_vec or quer_vec is None:
            return None

        suitable_doc = self._calculate_knn(quer_vec, self._document_vectors, n_neighbours)
        if not suitable_doc or suitable_doc is None:
            return None

        retrieve_res = []
        for item in suitable_doc:
            if not isinstance(item, tuple) or not isinstance(item[0], int):
                return None
            retrieve_res.append((item[1], self._documents[item[0]]))
        return retrieve_res

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
        if not isinstance(query_vector, tuple) or not query_vector:
            return None
        if len(query_vector) != len(self._document_vectors[0]):
            return None
        retrieve_doc = self._calculate_knn(query_vector, self._document_vectors, 1)
        if not isinstance(retrieve_doc, list) or not retrieve_doc:
            return None
        index, _ = retrieve_doc[0]
        return self._documents[index]

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
        if (not isinstance(document_vectors, list)
                or not isinstance(n_neighbours, int)
                or not query_vector or not document_vectors):
            return None

        neighbours = []
        for ind, val in enumerate(document_vectors):
            dist = calculate_distance(query_vector, val)
            if dist is None:
                return None
            neighbours.append((ind, dist))
        neighbours = sorted(neighbours, key=lambda x: x[1])
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
        if not isinstance(document, str) or not document:
            return None

        tok_doc = self._tokenizer.tokenize(document)
        if tok_doc is None:
            return None

        vec_doc = self._vectorizer.vectorize(tok_doc)
        if vec_doc is None:
            return None

        return vec_doc

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
        inf = [{
            'cur': [(vector, index) for index, vector in enumerate(vectors)],
            'depth': 0,
            'ancestor': Node(tuple([0.0] * len(vectors[0])), -1),
            'left': True
                }]

        while inf:
            cur_vect = inf[0]['cur']
            depth = inf[0]['depth']
            parent = inf[0]['ancestor']
            is_left = inf[0]['left']
            inf.pop(0)
            if cur_vect:
                axis = depth % len(cur_vect[0])
                cur_vect.sort(key=lambda vector: vector[0][axis])
                median_index = len(cur_vect) // 2
                median_node = Node(cur_vect[median_index][0],
                                   cur_vect[median_index][1])

                if parent.payload != -1 and is_left:
                    parent.left_node = median_node
                elif parent.payload == -1:
                    self._root = median_node
                else:
                    parent.right_node = median_node

                inf.append(
                    {
                    'cur': cur_vect[:median_index],
                    'depth': depth + 1,
                    'ancestor': median_node,
                    'left': True
                    })
                inf.append(
                    {
                    'cur': cur_vect[median_index + 1:],
                    'depth': depth + 1,
                    'ancestor': median_node,
                    'left': False
                    })
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
        if not isinstance(vector, tuple) or not isinstance(k, int):
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
        if not isinstance(vector, tuple) or not isinstance(k, int) or not vector:
            return None

        spaces = []
        spaces.append((self._root, 0))
        while spaces:
            node, depth = spaces.pop(0)
            if node is None:
                return None
            if node.left_node is None and node.right_node is None:
                distance = calculate_distance(vector, node.vector)
                return [(distance, node.payload)] if distance is not None else None
            axis = depth % len(node.vector)
            new_depth = depth + 1
            if vector[axis] <= node.vector[axis]:
                spaces.append((node.left_node, new_depth))
            else:
                spaces.append((node.right_node, new_depth))
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
        self._tree = NaiveKDTree()
        BasicSearchEngine.__init__(self, vectorizer = vectorizer, tokenizer = tokenizer)

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for retriever.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if document is successfully indexed

        In case of corrupt input arguments, False is returned.
        """
        if (not isinstance(documents, list) or not
        all(isinstance(item, str) for item in documents) or not documents):
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

        pre_query = self._index_document(query)

        if pre_query is None:
            return None

        res = self._tree.query(pre_query)

        if (res is None or not res
                or not all(isinstance(dist, float)
                           or isinstance(ind, int) for dist, ind in res)):
            return None

        result = []
        for dist, doc in res:
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
