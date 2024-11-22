"""
Lab 3.

Vector search with text retrieving
"""
from json import dump, load
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
    if document_vector is None or query_vector is None:
        return None
    if not query_vector or not document_vector:
        return 0.0
    summary = 0.0
    for i, value in enumerate(document_vector):
        summary += (query_vector[i] - value) ** 2
    distance = summary ** 0.5
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
    state_vector = {'len': len(vector), 'elements': {}}
    for i, value in enumerate(vector):
        if value == 0.0 or not isinstance(state_vector['elements'], dict):
            continue
        state_vector['elements'][i] = value
    return state_vector

def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): State of the vector to load from

    Returns:
        Vector | None: Loaded vector

    In case of corrupt input arguments, None is returned.
    """
    if 'len' not in state or 'elements' not in state:
        return None
    vector = [0.0] * state['len']
    for i in range(0, state['len']):
        if f'{i}' in state['elements']:
            vector[i] = state['elements'][f'{i}']
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
        for symbol in text:
            if symbol != ' ' and not symbol.isalpha():
                text = text.lower().replace(symbol, ' ')
        return self._remove_stop_words(text.split())

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(documents, list) or not documents:
            return None
        clean_docs = []
        for doc in documents:
            if not isinstance(doc, str):
                return None
            tokens = self.tokenize(doc)
            if tokens is None:
                return None
            clean_docs.append(tokens)
        return clean_docs

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(tokens, list) or not tokens:
            return None
        clean_tokens = []
        for token in tokens:
            if not isinstance(token, str):
                return None
            if token not in self._stop_words:
                clean_tokens.append(token)
        return clean_tokens


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
        vocab = set()
        if not isinstance(self._corpus, list):
            return False
        for doc in self._corpus:
            if not isinstance(doc, list) or not doc:
                return False
            vocab |= set(doc)
        sorted_vocab = sorted(vocab)
        self._vocabulary = sorted_vocab
        for i, token in enumerate(self._vocabulary):
            self._token2ind[token] = i
        idf = calculate_idf(self._vocabulary, self._corpus)
        if not isinstance(idf, dict):
            return False
        self._idf_values = idf
        if None in self._corpus or not self._vocabulary or not self._idf_values:
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
        if not self._vocabulary:
            return ()
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
        word_vector = []
        for i, value in enumerate(vector):
            if value == 0:
                continue
            word_vector.append(self._vocabulary[i])
        if not isinstance(word_vector, list):
            return None
        return word_vector

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance should be saved

        Returns:
            bool: True if saved successfully, False in other case
        """
        state = {'idf_values': self._idf_values,
                 'vocabulary': self._vocabulary,
                 'token2ind': self._token2ind}
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
        with open(file_path, 'r', encoding='utf-8') as file:
            state = load(file)
            if 'idf_values' not in state or 'vocabulary' not in state\
                or 'token2ind' not in state:
                return False
            self._idf_values = state['idf_values']
            self._vocabulary = state['vocabulary']
            self._token2ind = state['token2ind']
        if not self._idf_values or not self._vocabulary or not self._token2ind:
            return False
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
        zero_vector = (0.0, ) * len(self._vocabulary)
        tf = calculate_tf(self._vocabulary, document)
        if not isinstance(tf, dict):
            return None
        sorted_tf = sorted(tf.keys())
        tf_idf = {}
        for word in sorted_tf:
            if word not in self._vocabulary:
                continue
            tf_idf[word] = tf[word] * self._idf_values[word]
        values_tf_idf = list(tf_idf.values())
        list_zero_vector = list(zero_vector)
        for i, value in enumerate(values_tf_idf):
            list_zero_vector[i] = value
        return tuple(list_zero_vector)


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
        if not isinstance(documents, list) or not documents:
            return False
        self._documents = documents
        for text in documents:
            if not isinstance(text, str):
                return False
            vector = self._index_document(text)
            if not isinstance(vector, tuple):
                return False
            self._document_vectors.append(vector)
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
        vector_query = self._index_document(query)
        if not isinstance(vector_query, tuple):
            return None
        relevant_doc = self._calculate_knn(vector_query, self._document_vectors, 2)
        if not isinstance(relevant_doc, list) or not relevant_doc:
            return None
        text_rel_doc = []
        for couple in relevant_doc:
            index, value = couple
            if value is None:
                return None
            text_rel_doc.append((value, self._documents[index]))
        return text_rel_doc

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
        doc = self._documents[retrieve_doc[0][0]]
        return doc

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
        if n_neighbours <= 0 or not query_vector or not document_vectors:
            return None
        neighbours = {}
        for i, vector in enumerate(document_vectors):
            neighbour = calculate_distance(query_vector, vector)
            if not isinstance(neighbour, float):
                return None
            neighbours[i] = neighbour
        sorted_nbrs = sorted(neighbours.items(), key=lambda item: item[1])
        return sorted_nbrs[:n_neighbours]

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
        if not isinstance(tokenized_doc, list):
            return None
        vector_doc = self._vectorizer.vectorize(tokenized_doc)
        if not isinstance(vector_doc, tuple):
            return None
        return vector_doc

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
        i_vectors = []
        for i, vector in enumerate(vectors):
            i_vectors.append((vector, i))
        space = [(i_vectors, 0, Node((), -1), True)]
        while space:
            ind_vectors, depth, root, side_left = space.pop(0)
            if ind_vectors:
                axis = depth % len(ind_vectors[0])
                ind_vectors.sort(key = lambda x: x[0][axis])
                median_index = len(ind_vectors) // 2
                median = Node(ind_vectors[median_index][0], ind_vectors[median_index][1])
                if root.payload == -1:
                    self._root = median
                elif root.payload > -1 and side_left is True:
                    root.left_node = median
                else:
                    root.right_node = median
                space.append((ind_vectors[:median_index],
                              depth + 1,
                              median,
                              True))
                space.append((ind_vectors[median_index + 1:],
                              depth + 1,
                              median,
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
        if not isinstance(vector, tuple) or not vector\
            or not isinstance(k, int):
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
        if not isinstance(vector, tuple) or not vector\
            or not isinstance(k, int):
            return None
        root = [(self._root, 0)]
        while root:
            node = root[0][0]
            depth = root[0][1]
            root.pop(0)
            if node is None:
                return None
            if node.right_node is None and node.left_node is None:
                distance = calculate_distance(vector, node.vector)
                if distance is not None:
                    return [(distance, node.payload)]
                return None
            axis = depth % len(node.vector)
            new_depth = depth + 1
            if vector[axis] <= node.vector[axis]:
                root.append((node.left_node, new_depth))
            else:
                root.append((node.right_node, new_depth))
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
        if not isinstance(documents, list) or not documents:
            return False
        super().index_documents(documents)
        if self._tree.build(self._document_vectors):
            return True
        return False

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
        if not isinstance(query, str) or not query\
            or not isinstance(n_neighbours, int)\
            or not n_neighbours:
            return None
        vector_query = super()._index_document(query)
        if not isinstance(vector_query, tuple):
            return None
        relev_vectors = self._tree.query(vector_query)
        if relev_vectors is None or not relev_vectors:
            return None
        relev_docs = []
        for distance, index in relev_vectors:
            if distance is None or index is None:
                return None
            relev_docs.append((distance, self._documents[index]))
        return relev_docs


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
