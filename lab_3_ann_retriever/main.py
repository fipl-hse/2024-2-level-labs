"""
Lab 3.

Vector search with text retrieving
"""

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from typing import Protocol


class NodeLike(Protocol):
    """Type alias for a tree node."""

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: state of the Node instance
        """

    def load(self, state: dict) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict): saved state of the Node.

        Returns:
            bool: True is loaded successfully, False in other cases.
        """


Vector = tuple[float, ...]
"Type alias for vector representation of a text."


def save_vector(vector: Vector) -> dict:
    """
    Prepare a vector for save.

    Args:
        vector (Vector): Vector to save.

    Returns:
        dict: A state of the vector to save.
    """


def load_vector(state: dict) -> Vector | None:
    """
    Load vector from state.

    Args:
        state (dict): state of the vector to load from.

    Returns:
        Vector | None: loaded vector.
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
            stop_words (list[str]): List with stop words.
        """
        self.stop_words = stop_words

    def tokenize(self, text: str) -> list[str] | None:
        """
        Tokenize the input text into lowercase words without punctuation, digits and other symbols.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[str] | None: A list of words from the text.

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(text, str) or len(text) < 1:
            return None

        for symbol in text:
            if symbol.isalpha() or symbol == " ":
                continue
            text = text.replace(symbol, " ")

        processed_text = self._remove_stop_words(text.lower().split())

        return processed_text

    def tokenize_documents(self, documents: list[str]) -> list[list[str]] | None:
        """
        Tokenize the input documents .

        Args:
            documents (list[str]): Documents to tokenize.

        Returns:
            list[list[str]] | None: A list of tokenized documents.

        In case of corrupt input arguments, None is returned.
        """
        if (not isinstance(documents, list) or len(documents) < 1
                or not all(isinstance(doc, str) for doc in documents)):
            return None
        result = []
        for doc in documents:
            new_doc = self.tokenize(doc)
            if new_doc is None:
                return None
            result.append(new_doc)
        return result

    def _remove_stop_words(self, tokens: list[str]) -> list[str] | None:
        """
        Remove stopwords from the list of tokens.

        Args:
            tokens (list[str]): List of tokens.

        Returns:
            list[str] | None: Tokens after removing stopwords.

        In case of corrupt input arguments, None is returned.
        """
        if tokens is None or not isinstance(tokens, list) or not isinstance(self.stop_words, list):
            return None
        for token in tokens:
            if not isinstance(token, str):
                return None
        for word in self.stop_words:
            if not isinstance(word, str):
                return None
        if len(tokens) == 0 or len(self.stop_words) == 0:
            return None

        return list(filter(lambda x: x not in self.stop_words, tokens))


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
            corpus (list[list[str]]): Tokenized documents to vectorize.
        """

    def build(self) -> None:
        """
        Builds vocabulary with tokenized_documents.
        """

    def vectorize(self, tokenized_document: list[str]) -> Vector | None:
        """
        Create a vector for tokenized document.

        Args:
            tokenized_document (list[str]): Tokenized document to vectorize.

        Returns:
            Vector | None: TF-IDF vector for document.

        In case of corrupt input arguments, None is returned.
        """

    def _calculate_tf_idf(self, document: list[str]) -> Vector | None:
        """
        Getting TF-IDF for document.

        Args:
            document (list[str]): Tokenized document to vectorize.

        Returns:
            Vector | None: TF-IDF vector for document.

        In case of corrupt input arguments, None is returned.
        """

    def vector2tokens(self, vector: Vector) -> list[str] | None:
        """
        Recreate a tokenized document based on a vector.

        Args:
            vector (Vector): Vector to decode.

        Returns:
            list[str] | None: Tokenized document.

        In case of corrupt input arguments, None is returned.
        """

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance will be saved.

        Returns:
            bool: True if saved successfully, False in other case
        """

    def load(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance will be saved.

        Returns:
            bool: True if the vectorizer saved successfully.

        In case of corrupt input arguments, False is returned.
        """


def calculate_distance(query_vector: Vector, document_vector: Vector) -> float | None:
    """
    Calculate Euclidean distance for a document vector.

    Args:
        query_vector (Vector): Vectorized vector.
        document_vector (Vector): Vectorized documents.

    Returns:
        float | None: Euclidean distance for vector.

    In case of corrupt input arguments, None is returned.
    """


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
            vectorizer (Vectorizer): Vectorizer for documents vectorization.
            tokenizer (Tokenizer): Tokenizer for tokenization.
        """

    def _index_document(self, document: str) -> Vector | None:
        """
        Indexes document.

        Args:
            document (str): Document to index.

        Returns:
            Vector | None: Returns document vector.

        In case of corrupt input arguments, None is returned.
        """

    def index_documents(self, documents: list[str]) -> bool:
        """
        Indexes documents for engine.

        Args:
            documents (list[str]): Documents to index.

        Returns:
            bool: Returns True if documents are successfully indexed.

        In case of corrupt input arguments, False is returned.
        """

    def retrieve_relevant_documents(
        self, query: str, n_neighbours: int
    ) -> list[tuple[float, str]] | None:
        """
        Indexes documents for retriever.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]] | None: Relevant documents with their distances.

        In case of corrupt input arguments, None is returned.
        """

    def _calculate_knn(
        self, query_vector: Vector, document_vectors: list[Vector], n_neighbours: int
    ) -> list[tuple[int, float]] | None:
        """
        Calculate TF-IDF scores for a document.

        Args:
            query_vector (Vector): Vectorized vector.
            document_vectors (list[Vector]): Vectorized documents.
            n_neighbours (int): Number of neighbours to return.

        Returns:
            list[tuple[int, float]] | None: Nearest neighbours indices and distances.

        In case of corrupt input arguments, None is returned.
        """

    def _dump_documents(self) -> dict:
        """
        Dump documents states for save the Engine.

        Returns:
            dict: document and document_vectors states.
        """

    def save(self, file_path: str) -> bool:
        """
        Save the Vectorizer state to file.

        Args:
            file_path (str): The path to the file where the instance will be saved.

        Returns:
            bool: returns True if save was done correctly, False in another cases.
        """

    def _load_documents(self, state: dict) -> bool:
        """
        Load documents from state.

        Args:
            state (dict): state with documents.

        Returns:
            bool: True if documents was loaded, False in other cases.
        """

    def load(self, file_path: str) -> bool:
        """
        Load engine from state.

        Args:
            file_path (str): state with documents.

        Returns:
            bool: True if documents was loaded, False in other cases.
        """

    def retrieve_vectorized(self, query_vector: Vector) -> str | None:
        """
        Retrieve document by vector.

        Args:
            query_vector (Vector): Question vector.

        Returns:
            str | None: Answer document.

        In case of corrupt input arguments, None is returned.
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
            vector (Vector): Current vector node.
            payload (int): Index of current vector.
            left_node (NodeLike | None): Left node.
            right_node (NodeLike | None): Right node.
        """

    def save(self) -> dict:
        """
        Save Node instance to state.

        Returns:
            dict: state of the Node instance
        """

    def load(self, state: dict) -> bool:
        """
        Load Node instance from state.

        Args:
            state (dict): saved state of the Node.

        Returns:
            bool: True is loaded successfully, False in other cases.
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

    def build(self, vectors: list[Vector]) -> bool:
        """
        Build tree by calling _build_tree method.

        Args:
            vectors (list[Vector]): Vectors for tree building.

        Returns:
            bool: True if tree was built, False in other cases.

        In case of corrupt input arguments, False is returned.
        """

    def query(self, vector: Vector, k: int = 1) -> list[tuple[float, int]] | None:
        """
        Get k nearest neighbours for vector.

        Args:
            vector (Vector): Vector to get k nearest neighbours.
            k (int): Number of nearest neighbours to get.

        Returns:
            list[tuple[float, int]] | None: Nearest neighbours indices.

        In case of corrupt input arguments, None is returned.
        """

    def _find_closest(self, vector: Vector, k: int = 1) -> list[tuple[float, int]] | None:
        """
        Get k nearest neighbours for vector by filling best list.

        Args:
            vector (Vector): Vector for getting knn.
            k (int): The number of nearest neighbours to return.

        Returns:
            list[tuple[float, int]] | None: The list of k nearest neighbours.

        In case of corrupt input arguments, None is returned.
        """

    def load(self, state: dict) -> bool:
        """
        Load NaiveKDTree instance from state.

        Args:
            state (dict): saved state of the NaiveKDTree.

        Returns:
            bool: True is loaded successfully, False in other cases.
        """

    def save(self) -> dict | None:
        """
        Save NaiveKDTree instance to state.

        Returns:
            dict | None: state of the NaiveKDTree instance
        In case of corrupt input arguments, None is returned.
        """


class KDTree(NaiveKDTree):
    """
    KDTree.
    """

    def _find_closest(self, vector: Vector, k: int = 1) -> list[tuple[float, int]] | None:
        """
        Get k nearest neighbours for vector by filling best list.

        Args:
            vector (Vector): Vector for getting knn.
            k (int): The number of nearest neighbours to return.

        Returns:
            list[tuple[float, int]] | None: The list of k nearest neighbours.

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
            vectorizer (Vectorizer): Vectorizer for documents vectorization.
            tokenizer (Tokenizer): Tokenizer for tokenization.
        """

    def index_documents(self, documents: list[str]) -> bool:
        """
        Indexes documents for retriever.

        Args:
            documents (list[str]): Documents to index.

        Returns:
            bool: Returns True if document is successfully indexed.

        In case of corrupt input arguments, False is returned.
        """

    def retrieve_relevant_documents(
        self, query: str, n_neighbours: int = 1
    ) -> list[tuple[float, str]] | None:
        """
        Indexes documents for retriever.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]] | None: Relevant documents with their distances.

        In case of corrupt input arguments, None is returned.
        """

    def save(self, file_path: str) -> bool:
        """
        Save the SearchEngine instance to a file.

        Args:
            file_path (str): The path to the file where the instance should be saved.

        Returns:
            bool: True if saved successfully, False in other case
        """

    def load(self, file_path: str) -> bool:
        """
        Load a SearchEngine instance from a file.

        Args:
            file_path (str): The path to the file from which to load the instance.

        Returns:
            bool: True if engine was load successfully, False in other cases.
        """


class AdvancedSearchEngine(SearchEngine):
    """
    Retriever based on KDTree algorithm with priority.
    """

    _tree: KDTree

    def __init__(self, vectorizer: Vectorizer, tokenizer: Tokenizer) -> None:
        """
        Initialize an instance of the AdvancedSearchEngine class.

        Args:
            vectorizer (Vectorizer): Vectorizer for documents vectorization.
            tokenizer (Tokenizer): Tokenizer for tokenization.
        """
