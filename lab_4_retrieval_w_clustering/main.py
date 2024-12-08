"""
Lab 4.

Vector search with clusterization
"""

from lab_2_retrieval_w_bm25.main import calculate_bm25

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called
from lab_3_ann_retriever.main import (
    BasicSearchEngine,
    calculate_distance,
    Tokenizer,
    Vector,
    Vectorizer,
)

Corpus = list[str]
"Type alias for corpus of texts."
TokenizedCorpus = list[list[str]]
"Type alias for tokenized texts."


def get_paragraphs(text: str) -> list[str]:
    """
    Split text to paragraphs.

    Args:
        text (str): Text to split in paragraphs.

    Raises:
        ValueError: In case of inappropriate type input argument or if input argument is empty.

    Returns:
        list[str]: Paragraphs from document.
    """
    if not text:
        raise ValueError("Input argument is empty")
    if not isinstance(text, str):
        raise ValueError("Inappropriate type input argument")
    return text.split('\n')


class BM25Vectorizer(Vectorizer):
    """
    BM25 Vectorizer.
    """

    _corpus: TokenizedCorpus
    _avg_doc_len: float

    def __init__(self) -> None:
        """
        Initialize an instance of the BM25Vectorizer class.
        """
        self._corpus = []
        super().__init__(self._corpus)
        self._avg_doc_len = -1.0

    def set_tokenized_corpus(self, tokenized_corpus: TokenizedCorpus) -> None:
        """
        Set tokenized corpus and average document length.

        Args:
            tokenized_corpus (TokenizedCorpus): Tokenized texts corpus.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not tokenized_corpus:
            raise ValueError("Input argument is empty")
        if not isinstance(tokenized_corpus, list):
            raise ValueError("Inappropriate type input argument")
        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(map(len, tokenized_corpus))/len(tokenized_corpus)

    def vectorize(self, tokenized_document: list[str]) -> Vector:
        """
        Create a vector for tokenized document.

        Args:
            tokenized_document (list[str]): Tokenized document to vectorize.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if methods used return None.

        Returns:
            Vector: BM25 vector for document.
        """
        if not tokenized_document:
            raise ValueError("Input argument is empty")
        if not isinstance(tokenized_document, list):
            raise ValueError("Inappropriate type input argument")
        bm25 = self._calculate_bm25(tokenized_document)
        if bm25 is None:
            raise ValueError("Method returned None")
        return bm25

    def _calculate_bm25(self, tokenized_document: list[str]) -> Vector:
        """
        Get BM25 vector for tokenized document.

        Args:
            tokenized_document (list[str]): Tokenized document to vectorize.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.

        Returns:
            Vector: BM25 vector for document.
        """
        if not tokenized_document:
            raise ValueError("Input argument is empty")
        if not isinstance(tokenized_document, list):
            raise ValueError("Inappropriate type input argument")
        if not self._vocabulary:
            return ()
        vector_bm25 = [0.0]*len(self._vocabulary)
        dict_bm25 = calculate_bm25(self._vocabulary, tokenized_document, self._idf_values,
                                   avg_doc_len=self._avg_doc_len, doc_len=len(tokenized_document))
        if dict_bm25 is None:
            raise ValueError("Method returned None")
        for token in dict_bm25:
            if token in self._vocabulary:
                vector_bm25[self._token2ind[token]] = dict_bm25[token]
        return tuple(vector_bm25)


class DocumentVectorDB:
    """
    Document and vector database.
    """

    __vectors: dict[int, Vector]
    __documents: Corpus
    _tokenizer: Tokenizer
    _vectorizer: BM25Vectorizer

    def __init__(self, stop_words: list[str]) -> None:
        """
        Initialize an instance of the DocumentVectorDB class.

        Args:
            stop_words (list[str]): List with stop words.
        """
        self._tokenizer = Tokenizer(stop_words)
        self._vectorizer = BM25Vectorizer()
        self.__documents = []
        self.__vectors = {}

    def put_corpus(self, corpus: Corpus) -> None:
        """
        Fill documents and vectors based on corpus.

        Args:
            corpus (Corpus): Corpus of texts.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if methods used return None.
        """
        if not corpus:
            raise ValueError("Input argument is empty")
        if not isinstance(corpus, list):
            raise ValueError("Inappropriate type input argument")
        documents = []
        vectors = {}
        corpus_tokenized = []
        for text in corpus:
            text_tokenized = self._tokenizer.tokenize(text)
            if text_tokenized is None:
                raise ValueError("Method returned None")
            if text_tokenized:
                corpus_tokenized.append(text_tokenized)
                documents.append(text)
        self._vectorizer.set_tokenized_corpus(corpus_tokenized)
        self._vectorizer.build()
        i = 0
        for text_tokenized in corpus_tokenized:
            text_vectorized = self._vectorizer.vectorize(text_tokenized)
            if text_vectorized is None:
                raise ValueError("Method returned None")
            vectors[i] = text_vectorized
            i += 1
        self.__documents = documents
        self.__vectors = vectors

    def get_vectorizer(self) -> BM25Vectorizer:
        """
        Get an object of the BM25Vectorizer class.

        Returns:
            BM25Vectorizer: BM25Vectorizer class object.
        """
        return self._vectorizer

    def get_tokenizer(self) -> Tokenizer:
        """
        Get an object of the Tokenizer class.

        Returns:
            Tokenizer: Tokenizer class object.
        """
        return self._tokenizer

    def get_vectors(self, indices: list[int] | None = None) -> list[tuple[int, Vector]]:
        """
        Get document vectors by indices.

        Args:
            indices (list[int] | None): Document indices.

        Returns:
            list[tuple[int, Vector]]: List of index and vector for documents.
        """
        if indices is None:
            return list(self.__vectors.items())
        return [(index, self.__vectors[index]) for index in indices]

    def get_raw_documents(self, indices: tuple[int, ...] | None = None) -> Corpus:
        """
        Get documents by indices.

        Args:
            indices (tuple[int, ...] | None): Document indices.

        Raises:
            ValueError: In case of inappropriate type input argument.

        Returns:
            Corpus: List of documents.
        """
        if indices is None:
            return self.__documents
        return [self.__documents[index] for index in sorted(set(indices), reverse=True)]


class VectorDBSearchEngine(BasicSearchEngine):
    """
    Engine based on VectorDB.
    """

    _db: DocumentVectorDB

    def __init__(self, db: DocumentVectorDB) -> None:
        """
        Initialize an instance of the RerankerEngine class.

        Args:
            db (DocumentVectorDB): Object of DocumentVectorDB class.
        """
        self._db = db
        super().__init__(self._db.get_vectorizer(), self._db.get_tokenizer())

    def retrieve_relevant_documents(self, query: str, n_neighbours: int) -> list[tuple[float, str]]:
        """
        Get relevant documents.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]]: Relevant documents with their distances.
        """
        if not query or not n_neighbours:
            raise ValueError("Input argument is empty")
        if not isinstance(n_neighbours, int):
            raise ValueError("Inappropriate type input argument")
        if n_neighbours < 0:
            raise ValueError("Inappropriate value input argument")
        document_vectors = [vectors[1] for vectors in self._db.get_vectors()]
        query_vectorized = self._index_document(query)
        if not isinstance(query_vectorized, tuple):
            raise ValueError("Method returned None")
        knn = self._calculate_knn(query_vectorized, document_vectors, n_neighbours)
        if not isinstance(knn, list):
            raise ValueError("Method returned None")
        document_indexes = tuple([vectors[0]][0] for vectors in knn)
        documents_raw = self._db.get_raw_documents(document_indexes)
        relevant_documents = [(document[1], documents_raw[index])
                              for index, document in enumerate(knn)]
        return relevant_documents


class ClusterDTO:
    """
    Store clusters.
    """

    __centroid: Vector
    __indices: list[int]

    def __init__(self, centroid_vector: Vector) -> None:
        """
        Initialize an instance of the ClusterDTO class.

        Args:
            centroid_vector (Vector): Centroid vector.
        """
        self.__centroid = centroid_vector
        self.__indices = []

    def __len__(self) -> int:
        """
        Return the number of document indices.

        Returns:
            int: The number of document indices.
        """
        return len(self.__indices)

    def get_centroid(self) -> Vector:
        """
        Get cluster centroid.

        Returns:
            Vector: Centroid of current cluster.
        """
        return self.__centroid

    def set_new_centroid(self, new_centroid: Vector) -> None:
        """
        Set new centroid for cluster.

        Args:
            new_centroid (Vector): New centroid vector.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty.
        """
        if not new_centroid or new_centroid is None:
            raise ValueError("Input argument is empty")
        if not isinstance(new_centroid, tuple):
            raise ValueError("Inappropriate type input argument")
        self.__centroid = new_centroid

    def erase_indices(self) -> None:
        """
        Clear indexes.
        """
        self.__indices = []

    def add_document_index(self, index: int) -> None:
        """
        Add document index.

        Args:
            index (int): Index of document.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty.
        """
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Inappropriate type input argument")
        if not index:
            raise ValueError("Input argument is empty")
        if index < 0:
            raise ValueError("Inappropriate value input argument")
        if index not in self.__indices:
            self.__indices.append(index)

    def get_indices(self) -> list[int]:
        """
        Get indices.

        Returns:
            list[int]: Indices of documents.
        """
        return self.__indices


class KMeans:
    """
    Train k-means algorithm.
    """

    __clusters: list[ClusterDTO]
    _db: DocumentVectorDB
    _n_clusters: int

    def __init__(self, db: DocumentVectorDB, n_clusters: int) -> None:
        """
        Initialize an instance of the KMeans class.

        Args:
            db (DocumentVectorDB): An instance of DocumentVectorDB class.
            n_clusters (int): Number of clusters.
        """
        self._db = db
        self.__clusters = []
        self._n_clusters = n_clusters

    def train(self) -> None:
        """
        Train k-means algorithm.
        """
        vectors_initial = self._db.get_vectors(list(range(self._n_clusters)))
        self.__clusters = [ClusterDTO(vector[1]) for vector in vectors_initial]
        while not self._is_convergence_reached(self.__clusters):
            self.run_single_train_iteration()

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        centroids = []
        for cluster in self.__clusters:
            cluster.erase_indices()
            centroids.append(cluster.get_centroid())
        vectors = self._db.get_vectors()
        for vector_index, vector in vectors:
            closest_centroid = centroids[0]
            distance_minimal = calculate_distance(vector, centroids[0])
            if distance_minimal is None:
                raise ValueError("Input argument is empty")
            for centroid in centroids:
                distance = calculate_distance(vector, centroid)
                if distance is None:
                    raise ValueError("Input argument is empty")
                if distance < distance_minimal:
                    distance_minimal = distance
                    closest_centroid = centroid
            self.__clusters[centroids.index(closest_centroid)].add_document_index(vector_index)
        for cluster in self.__clusters:
            cluster_vectors = [vectors[index][1] for index in cluster.get_indices()]
            centroid_updated = (sum(cluster_vectors[index]) / len(cluster_vectors)
                                for index in range(len(cluster_vectors)-1))
            cluster.set_new_centroid(tuple(centroid_updated))
        return self.__clusters

    def infer(self, query_vector: Vector, n_neighbours: int) -> list[tuple[float, int]]:
        """
        Launch clustering model inference.

        Args:
            query_vector (Vector): Vector of query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if input arguments are incorrect,
                or if methods used return None.

        Returns:
            list[tuple[float, int]]: Distance to relevant document and document index.
        """
        if not isinstance(query_vector, tuple) or not isinstance(n_neighbours, int):
            raise ValueError("Inappropriate type input argument")
        if not query_vector:
            raise ValueError("Input argument is empty")
        if n_neighbours < 1:
            raise ValueError("Inappropriate value input argument")
        centroids = [cluster.get_centroid() for cluster in self.__clusters]
        closest_centroid = centroids[0]
        distance_minimal = calculate_distance(query_vector, centroids[0])
        if distance_minimal is None:
            raise ValueError("Method returned None")
        for centroid in centroids:
            distance = calculate_distance(query_vector, centroid)
            if distance is None:
                raise ValueError("Method returned None")
            if distance < distance_minimal:
                distance_minimal = distance
                closest_centroid = centroid
        closest_index = centroids.index(closest_centroid)
        closest_cluster = self.__clusters[closest_index]
        indices = closest_cluster.get_indices()
        vectors = self._db.get_vectors(indices)
        documents_relevant = []
        distance_vector_minimal = calculate_distance(query_vector, centroids[0])
        if distance_vector_minimal is None:
            raise ValueError("Method returned None")
        for index, vector in vectors:
            distance_vector = calculate_distance(query_vector, vector)
            if distance_vector is None:
                raise ValueError("Method returned None")
            documents_relevant.append((distance_vector, index))
        documents_relevant.sort(key=lambda x: x[0])
        return documents_relevant[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int| list[str]]]: List with information about each cluster
        """

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """

    def _is_convergence_reached(
        self, new_clusters: list[ClusterDTO], threshold: float = 1e-07
    ) -> bool:
        """
        Check the convergence of centroids.

        Args:
            new_clusters (list[ClusterDTO]): Centroids after updating.
            threshold (float): Threshold for determining the distance correctness.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if methods used return None.

        Returns:
            bool: True if the distance is correct, False in other cases.
        """
        if not self.__clusters:
            raise ValueError("Input argument is empty")
        for index, old_cluster in enumerate(self.__clusters):
            new_centroid = new_clusters[index].get_centroid()
            old_centroid = old_cluster.get_centroid()
            distance = calculate_distance(new_centroid, old_centroid)
            if distance is None:
                raise ValueError("Method returned None")
            if distance > threshold:
                return False
        return True


class ClusteringSearchEngine:
    """
    Engine based on KMeans algorithm.
    """

    __algo: KMeans
    _db: DocumentVectorDB

    def __init__(self, db: DocumentVectorDB, n_clusters: int = 3) -> None:
        """
        Initialize an instance of the ClusteringSearchEngine class.

        Args:
            db (DocumentVectorDB): An instance of DocumentVectorDB class.
            n_clusters (int): Number of clusters.
        """

    def retrieve_relevant_documents(self, query: str, n_neighbours: int) -> list[tuple[float, str]]:
        """
        Get relevant documents.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty,
                or if input arguments are incorrect,
                or if methods used return None.

        Returns:
            list[tuple[float, str]]: Relevant documents with their distances.
        """

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """

    def calculate_square_sum(self) -> float:
        """
        Get sum by all clusters of sum of squares of distance from vector of clusters to centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """


class VectorDBEngine:
    """
    Engine wrapper that encapsulates different engines and provides unified API to it.
    """

    _db: DocumentVectorDB
    _engine: BasicSearchEngine

    def __init__(self, db: DocumentVectorDB, engine: BasicSearchEngine) -> None:
        """
        Initialize an instance of the ClusteringSearchEngine class.

        Args:
            db (DocumentVectorDB): An instance of DocumentVectorDB class.
            engine (BasicSearchEngine): A search engine.
        """

    def retrieve_relevant_documents(
        self, query: str, n_neighbours: int
    ) -> list[tuple[float, str]] | None:
        """
        Index documents for retriever.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]] | None: Relevant documents with their distances.

        """


class VectorDBTreeSearchEngine(VectorDBEngine):
    """
    Engine provided unified interface to SearchEngine.
    """

    def __init__(self, db: DocumentVectorDB) -> None:
        """
        Initialize an instance of the VectorDBTreeSearchEngine class.

        Args:
            db (DocumentVectorDB): An instance of DocumentVectorDB class.
        """


class VectorDBAdvancedSearchEngine(VectorDBEngine):
    """
    Engine provided unified interface to AdvancedSearchEngine.
    """

    def __init__(self, db: DocumentVectorDB) -> None:
        """
        Initialize an instance of the VectorDBAdvancedSearchEngine class.

        Args:
            db (DocumentVectorDB): An instance of DocumentVectorDB class.
        """
