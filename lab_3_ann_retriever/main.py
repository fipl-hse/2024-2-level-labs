"""
Lab 3.

Vector search with text retrieving
"""

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
    if (not isinstance(query_vector, tuple) or not isinstance(document_vector, tuple)):
        return None
    distance = 0.0
    if len(query_vector) == 0 or len(document_vector) == 0:
        return distance
    for coord, value in enumerate(document_vector):
        distance += (query_vector[coord] - value) ** 2
    if not isinstance(distance, float):
        return None
    return float(distance ** 0.5)

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
        if not isinstance(text, str) or not isinstance(self, Tokenizer):
            return None
        return self._remove_stop_words(
            ''.join([letter for letter in text.lower()
                     if letter == ' ' or letter.isalpha()]).split()
        )

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents.

        Args:
            documents (list[str]): Documents to tokenize

        Returns:
            list[list[str]] | None: A list of tokenized documents

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(documents, list) or len(documents) == 0:
            return None
        unique_words = []
        for text in documents:
            if not isinstance(text, str):
                return None
            new_tokens = self.tokenize(text)
            if new_tokens is None:
                return None
            unique_words.append(new_tokens)
        return unique_words

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens

        Returns:
            list[str] | None: Tokens after removing stopwords

        In case of corrupt input arguments, None is returned.
        """
        if (not isinstance(tokens, list) or not isinstance(self, Tokenizer)
                or len(tokens) == 0):
            return None
        if tokens == [] or self._stop_words == []:
            return None
        for data in tokens:
            if not isinstance(data, str):
                return None
        for data in self._stop_words:
            if not isinstance(data, str):
                return None
        return [elem for elem in tokens if elem not in self._stop_words]

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

        if self._corpus is None or len(self._corpus) == 0:
            return False
        self._vocabulary = sorted(list(set(token for doc in self._corpus for token in doc)))

        text = calculate_idf(self._vocabulary, self._corpus)
        if text is None:
            return False
        self._idf_values = text

        for ind, token in enumerate(self._vocabulary):
            self._token2ind[token] = int(ind)
        if (None in self._vocabulary or None in self._idf_values
                or None in self._token2ind):
            return False
        if (len(self._vocabulary) == 0 or len(self._idf_values) == 0
            or len(self._token2ind) == 0):
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
        if not isinstance(tokenized_document, list) or len(tokenized_document) == 0:
            return None
        if len(self._vocabulary) == 0:
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
        if (not isinstance(vector, tuple) or len(vector) == 0
            or len(vector) != len(self._token2ind)):
            return None
        tokens = []
        tokens2ind_new = dict((v, k) for k, v in self._token2ind.items())
        for ind, coord in enumerate(vector):
            if coord != 0.0:
                token_current = tokens2ind_new.get(ind)
                if token_current is None:
                    return None
                tokens.append(token_current)
        return tokens
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
        if not isinstance(document, list) or len(document) == 0:
            return None
        vector = [0.0 for elem in self._vocabulary]
        text_tf = calculate_tf(self._vocabulary, document)
        if text_tf is None or len(text_tf) == 0:
            return None
        for token, ind in self._token2ind.items():
            vector[ind] = text_tf[token] * self._idf_values[token]
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

        if not isinstance(documents, list) or len(documents) == 0:
            return False
        self._documents = documents
        for text in documents:
            if not isinstance(text, str):
                return False
            vector = self._index_document(text)
            if vector is None or len(vector) == 0:
                return False
        docs_to_append = []
        for text in documents:
            doc_vector = self._index_document(text)
            if doc_vector is None:
                return False
            docs_to_append.append(doc_vector)
        self._document_vectors = docs_to_append
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
        if (not isinstance(query, str) or len(query) == 0
                or not isinstance(n_neighbours, int)):
            return None
        result = []
        vector_query = self._index_document(query)
        search_result = self._calculate_knn(vector_query,
        self._document_vectors, n_neighbours)
        if search_result is None or len(search_result) == 0:
            return None
        for data in search_result:
            if None in data:
                return None
            result.append((data[1], self._documents[data[0]]))
        return result
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
        if (not isinstance(query_vector, tuple) or len(query_vector) == 0
            or len(query_vector) != len(self._document_vectors[0])):
            return None
        answer = self._calculate_knn(query_vector, self._document_vectors, 1)
        if answer is None or len(answer) == 0 or None in answer:
            return None
        doc_ind = int(answer[0][0])
        return self._documents[doc_ind]
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
        if (not isinstance(query_vector, tuple) or not isinstance(document_vectors, list)
            or not isinstance(n_neighbours, int) or len(document_vectors) == 0):
            return None
        docs = []
        for ind, doc in enumerate(document_vectors):
            if not isinstance(doc, tuple) or len(doc) == 0:
                return None
            distance = calculate_distance(query_vector, doc)
            docs.append((int(ind), distance))
        return sorted(docs, key= lambda x:x[1])[:n_neighbours]

    def _index_document(self, document: str) -> Vector | None:
        """
        Index document.

        Args:
            document (str): Document to index

        Returns:
            Vector | None: Returns document vector

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(document, str) or len(document) == 0:
            return None
        tokenized_text = self._tokenizer.tokenize(document)
        return self._vectorizer.vectorize(tokenized_text)
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
        if not isinstance(vectors, list) or len(vectors) == 0:
            return False
        depth = 0
        vector_lst = {}
        for ind, vector in enumerate(vectors):
            vector_lst.update({vector: ind})
        dimensions = len(vectors[0])
        node_parent = Node()
        dimension_info = [(vectors,
                           depth,
                           node_parent,
                           True)]
        while len(dimension_info) != 0:
            dimension_info_copy = dimension_info.pop(0)
            if len(dimension_info_copy[0]) == 0:
                continue
            axis = int(int(dimension_info_copy[1]) % dimensions)
            dimension_vectors = sorted(dimension_info_copy[0], key=lambda x:x[axis])
            median_index = len(dimension_vectors) // 2
            node_vector = dimension_vectors[median_index]
            new_node = Node(node_vector, int(vector_lst[node_vector]))
            if dimension_info_copy[2].payload == -1:
                self._root = new_node
            elif dimension_info_copy[-1]:
                dimension_info_copy[2].left_node = new_node
            else:
                dimension_info_copy[2].right_node = new_node
            dimension_info.extend([(dimension_vectors[:median_index],
                                    dimension_info_copy[1] + 1,
                                    new_node,
                                    True),
                                  (dimension_vectors[median_index + 1:],
                                   dimension_info_copy[1] + 1,
                                   new_node,
                                   False)])
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
        if not isinstance(vector, tuple) or not isinstance(k, int) or len(vector) == 0 or k != 1:
            return None
        result = self._find_closest(vector, k)
        return result
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
        if not isinstance(vector, tuple) or not isinstance(k, int) or len(vector) == 0 or k != 1:
            return None
        depth = 0
        dimensions = len(self._root.vector)
        data = [[self._root,
                depth]]
        neighbours = []
        while True:
            data_copy = data.pop(0)
            current_node = data_copy[0]
            distance = calculate_distance(vector, current_node.vector)
            if distance is None:
                return None
            if current_node.left_node is None and current_node.right_node is None:
                neighbours.append((distance, current_node.payload))
                break
            axis = data_copy[1] % dimensions
            if current_node.left_node is not None and current_node.right_node is not None:
                if vector[axis] <= current_node.vector[axis]:
                    data.append([data_copy[0].left_node, data_copy[1] + 1])
                else:
                    data.append([data_copy[0].right_node, data_copy[1] + 1])
            elif data_copy[0].right_node is None:
                data.append([data_copy[0].left_node, data_copy[1] + 1])
            else:
                data.append([data_copy[0].right_node, data_copy[1] + 1])
        return sorted(neighbours)[:k]
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
        super().__init__(tokenizer= tokenizer, vectorizer= vectorizer)
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
        if not isinstance(documents, list) or len(documents) == 0:
            return False
        if not super().index_documents(documents):
            return False
        if not self._tree.build(self._document_vectors):
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
        if (not isinstance(query, str) or len(query) == 0
                or query is None or not isinstance(n_neighbours, int) or n_neighbours != 1):
            return None
        if not self.index_documents(self._documents):
            return None
        query_vector = super()._index_document(query)
        result = self._tree.query(query_vector, n_neighbours)
        if result is None or len(result) == 0:
            return None
        if None in result[0]:
            return None
        dist = result[0][0]
        ind = int(result[0][1])
        final = [(dist, self._documents[ind])]
        return final

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
        SearchEngine.__init__(self, vectorizer, tokenizer)
