"""
Lab 3.

Vector search with text retrieving
"""
import math
from lab_2_retrieval_w_bm25.main import calculate_idf, calculate_tf, calculate_tf_idf, tokenize
import math

# pylint: disable=too-few-public-methods, too-many-arguments, duplicate-code, unused-argument
from typing import Protocol

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
    if len(query_vector) != len(document_vector):
        return None
    calculation = 0.0
    for i in range(len(query_vector)):
        calculation += (query_vector[i] - document_vector[i]) ** 2
    return math.sqrt(calculation)


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
        if not isinstance(text, str):
            return None

        text = tokenize(text)
        return text


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
        list_for_tokens = []
        for string in documents:
            string = tokenize(string)
            list_for_tokens.append(string)
        return list_for_tokens


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
        self._vocabulary = []
        self._idf_values = {}
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
        for words in self._corpus:
            unique_words.update(words)
        if unique_words:
            self._vocabulary = sorted(unique_words)
            self._token2ind = {token: ind for ind, token in enumerate(self._vocabulary)}
            self._idf_values = calculate_idf(self._vocabulary, self._corpus)
            return True
        return False


    def vectorize(self, tokenized_document: list[str]) -> Vector | None:
        """
        Create a vector for tokenized document.

        Args:
            tokenized_document (list[str]): Tokenized document to vectorize

        Returns:
            Vector | None: TF-IDF vector for document

        In case of corrupt input arguments, None is returned.
        """
        if not isinstance(tokenized_document, list) or not tokenized_document \
                or not all(isinstance(word, str) for word in tokenized_document):
            return None
        if self._calculate_tf_idf(tokenized_document):
            return self._calculate_tf_idf(tokenized_document)
        return None


    def vector2tokens(self, vector: Vector) -> list[str] | None:
        """
        Recreate a tokenized document based on a vector.

        Args:
            vector (Vector): Vector to decode

        Returns:
            list[str] | None: Tokenized document

        In case of corrupt input arguments, None is returned.
        """
        if vector is None or not isinstance(vector, (list, tuple)):
            return None
        if len(vector) != len(self._token2ind):
            return None

        index_to_token = {index: token for pair in self._token2ind for index, token in pair}
        tokens = []
        for index, value in enumerate(vector):
            if value > 0:
                token = index_to_token.get(index)
                if token:
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
        if not isinstance(document, list) or not(all(isinstance(i, str) for i in document)):
            return None
        dict_with_zero = dict.fromkeys(self._vocabulary, 0.0)
        tf = calculate_tf(self._vocabulary, document)
        if tf is None:
            return None
        tf_idf = calculate_tf_idf(tf, self._idf_values)

        numbers = []
        for token in dict_with_zero:
            if token in tf_idf:
                dict_with_zero[token] = tf_idf[token]
        for key, value in dict_with_zero.items():
            numbers.append(value)
        return tuple(numbers)


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


    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for engine.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if documents are successfully indexed

        In case of corrupt input arguments, False is returned.
        """
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
            return False
        self._documents = documents
        list_with_vectorized_documents = []
        for doc in documents:
            vectorized_doc = self._index_document(doc)
            if not vectorized_doc:
                return False
            list_with_vectorized_documents.append(vectorized_doc)
        if list_with_vectorized_documents:
            self._document_vectors = list_with_vectorized_documents
            return True
        return False


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
        if not isinstance(query, str) or not isinstance(n_neighbours, int)\
                or isinstance(n_neighbours, bool):
            return None

        vector = self._index_document(query)
        if not vector:
            return None
        knn = self._calculate_knn(vector, self._document_vectors, n_neighbours)
        if not knn:
            return None
        result = []
        for tup in knn:
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
        #retrieve = self._vectorizer.vector2tokens(query_vector)


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
        if not document_vectors or not query_vector or not n_neighbours:
            return None
        if (document_vectors is None or n_neighbours <= 0 or
                n_neighbours > len(document_vectors) or not document_vectors or
                any(not isinstance(doc, (list, tuple)) for doc in document_vectors)):
            return None
        list_of_values = []
        for index, document_vector in enumerate(document_vectors):
            distance = calculate_distance(query_vector, document_vector)
            if distance is not None:
                list_of_values.append((index, distance))
        sorted_tuples = sorted(list_of_values, key=lambda number: number[1])
        return sorted_tuples[:n_neighbours]


    def _index_document(self, document: str) -> Vector | None:
        """
        Index document.

        Args:
            document (str): Document to index

        Returns:
            Vector | None: Returns document vector

        In case of corrupt input arguments, None is returned.
        """
        tokenized_doc = self._tokenizer.tokenize(document)
        if not tokenized_doc:
            return None
        vectorized_doc = self._vectorizer.vectorize(tokenized_doc)
        return vectorized_doc


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

vectorizer = Vectorizer([
            [
                "вектора",
                "используются",
                "поиска",
                "релевантного",
                "документа",
                "давайте",
                "научимся",
                "создавать",
                "использовать",
            ],[
                "кот",
                "вектор",
                "утрам",
                "приносит",
                "тапочки",
                "вечерам",
                "гуляем",
                "шлейке",
                "дворе",
                "вектор",
                "забавный",
                "храбрый",
                "боится",
                "собак",
            ],
            [
                "котёнок",
                "которого",
                "нашли",
                "дворе",
                "очень",
                "забавный",
                "пушистый",
                "утрам",
                "играю",
                "догонялки",
                "работой",
            ],
            [
                "собака",
                "думает",
                "любимый",
                "плед",
                "это",
                "кошка",
                "просто",
                "очень",
                "пушистый",
                "мягкий",
                "забавно",
                "наблюдать",
                "спят",
                "вместе"]])
tokenizer = Tokenizer(['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со',
                       'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
                       'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот',
                       'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда',
                       'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни',
                       'быть', 'был','него', 'до', 'вас', 'нибудь', 'опять', 'уж',
                       'вам', 'ведь', 'там', 'потом', 'себя', 'ничего','ей', 'может',
                       'они', 'тут', 'где', 'есть', 'надо', 'ней',  'для', 'мы', 'тебя',
                       'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'моя', 'это', 'очень', 'как'])
a = BasicSearchEngine(vectorizer, tokenizer)
print(a._index_document((
            "Моя собака думает, что ее любимый плед - это кошка. "
            "Просто он очень пушистый и мягкий. "
            "Забавно наблюдать, как они спят вместе! "
        )))

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

    def build(self, vectors: list[Vector]) -> bool:
        """
        Build tree.

        Args:
            vectors (list[Vector]): Vectors for tree building

        Returns:
            bool: True if tree was built, False in other cases

        In case of corrupt input arguments, False is returned.
        """

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

    def index_documents(self, documents: list[str]) -> bool:
        """
        Index documents for retriever.

        Args:
            documents (list[str]): Documents to index

        Returns:
            bool: Returns True if document is successfully indexed

        In case of corrupt input arguments, False is returned.
        """

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

