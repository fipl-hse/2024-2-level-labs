"""
Lab 4.

Vector search with clusterization
"""
import copy

from lab_2_retrieval_w_bm25.main import calculate_bm25, calculate_idf
from lab_3_ann_retriever.main import (
    AdvancedSearchEngine,
    BasicSearchEngine,
    calculate_distance,
    SearchEngine,
    Tokenizer,
    Vector,
    Vectorizer,
)

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called


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
    if not isinstance(text, str) or not text:
        raise ValueError(f'Incorrect value: "{text}" is not a string or empty.')

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
        if (not isinstance(tokenized_corpus, list) or
                not all(all(isinstance(clause, str) for clause in doc)
                        for doc in tokenized_corpus) or
                not tokenized_corpus):
            raise ValueError(f'Incorrect value: "{tokenized_corpus}" is not a list or empty.')

        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(len(doc) for doc in tokenized_corpus) / len(tokenized_corpus)

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
        if (not isinstance(tokenized_document, list) or
                not all(isinstance(clause, str) for clause in tokenized_document) or
                not tokenized_document):
            raise ValueError(f'Incorrect value: "{tokenized_document}" is not a list or empty.')

        bm25_document = self._calculate_bm25(tokenized_document)
        if bm25_document is None:
            raise ValueError(f'Incorrect value: "{bm25_document}" is None.')
        return bm25_document

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
        if (not isinstance(tokenized_document, list) or
                not all(isinstance(text, str) for text in tokenized_document) or
                not tokenized_document):
            raise ValueError(f'Incorrect value: "{tokenized_document}" is not a list or empty.')

        if not self._vocabulary or not self._corpus:
            return ()
        idf_document = calculate_idf(self._vocabulary, self._corpus)
        if idf_document is None:
            raise ValueError(f'Incorrect value: "{idf_document}" is empty, '
                             f'is None or is zero.')

        avg_doc_len = sum(len(doc) for doc in self._corpus) / len(self._corpus)
        doc_len = len(tokenized_document)
        k1 = 1.5
        b = 0.75
        bm25_dict = calculate_bm25(self._vocabulary, tokenized_document,
                                   idf_document, k1, b, avg_doc_len, doc_len)

        vector = [0.0] * len(self._vocabulary)
        for index, token in enumerate(self._vocabulary):
            if bm25_dict is not None and token in bm25_dict:
                vector[index] = bm25_dict[token]

        return tuple(vector)


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
        self.__vectors = {}
        self.__documents = []
        self._tokenizer = Tokenizer(stop_words)
        self._vectorizer = BM25Vectorizer()

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
        if (not isinstance(corpus, list) or
                not all(isinstance(text, str) for text in corpus) or
                not corpus):
            raise ValueError(f'Incorrect value: "{corpus}" is not a list or empty.')

        tokenized_corpus = []
        for doc in corpus:
            tokenized_doc = self._tokenizer.tokenize(doc)
            if tokenized_doc:
                tokenized_corpus.append(tokenized_doc)
                self.__documents.append(doc)

        self._vectorizer.set_tokenized_corpus(tokenized_corpus)
        self._vectorizer.build()

        for index, tokenized_doc in enumerate(tokenized_corpus):
            self.__vectors[index] = self._vectorizer.vectorize(tokenized_doc)

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

        required_documents = []
        for index in set(indices):
            required_documents.append(self.__documents[index])
        return required_documents


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
        super().__init__(db.get_vectorizer(), db.get_tokenizer())
        self._db = db

    def retrieve_relevant_documents(self, query: str, n_neighbours: int) -> list[tuple[float, str]]:
        """
        Get relevant documents.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]]: Relevant documents with their distances.
        """
        if (not query or
                not isinstance(query, str) or
                not isinstance(n_neighbours, int) or
                n_neighbours <= 0):
            raise ValueError(f'Incorrect value: "{query}" is not a string, empty, '
                             f'is None or is zero or '
                             f'"{n_neighbours}" is not an integer, '
                             f'is zero or is less than zero.')

        tokenized_query = self._tokenizer.tokenize(query)
        if not tokenized_query:
            raise ValueError(f'Incorrect value: "{tokenized_query}" is empty, '
                             f'is None or is zero.')

        query_vector = self._vectorizer.vectorize(tokenized_query)
        if not query_vector:
            raise ValueError(f'Incorrect value: "{query_vector}" is empty, '
                             f'is None or is zero.')

        vectors = [vector_data[1] for vector_data in self._db.get_vectors()]
        neighbours = self._calculate_knn(query_vector, vectors, n_neighbours)
        if not neighbours:
            raise ValueError(f'Incorrect value: "{neighbours}" is empty, '
                             f'is None or is zero.')

        relevant_documents = self._db.get_raw_documents(
                                            tuple(neighbor[0] for neighbor in neighbours))
        relevant_documents_with_distances = []
        for neighbor in neighbours:
            relevant_documents_with_distances.append((neighbor[1], relevant_documents[neighbor[0]]))
        return relevant_documents_with_distances


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
        if not new_centroid:
            raise ValueError(f'Incorrect value: "{new_centroid}" is empty, '
                             f'is None or is zero.')
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
        if (not isinstance(index, int) or index < 0):
            raise ValueError(f'Incorrect value: "{index}" is empty, is None, '
                             f'is not an integer or is less than zero.')

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
        self._n_clusters = n_clusters
        self.__clusters = []

    def train(self) -> None:
        """
        Train k-means algorithm.
        """
        if not self._db or not self._db.get_vectors():
            raise ValueError(f'Incorrect value: "{self._db}" is empty or does not exist.')

        vectors = self._db.get_vectors()[:self._n_clusters]
        if len(vectors) < self._n_clusters:
            raise ValueError(f'Incorrect value: length of the vectors ({len(vectors)}) '
                             f'less than number of the clusters ({self._n_clusters}).')

        self.__clusters = [ClusterDTO(vector[1]) for vector in vectors]
        while True:
            new_centroids = self.run_single_train_iteration()
            if self._is_convergence_reached(new_centroids):
                break
            self.__clusters = new_centroids

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
        for vector in vectors:
            distances_to_centroids = []
            for centroid in centroids:
                distance = calculate_distance(vector[1], centroid)
                if distance is None:
                    raise ValueError((f'Incorrect value: "{distance}" is empty, '
                                      f'is None or is zero.'))
                distances_to_centroids.append((distance, centroids.index(centroid)))

            min_cluster_index = min(distances_to_centroids)[1]
            self.__clusters[min_cluster_index].add_document_index(vectors.index(vector))

        for cluster in self.__clusters:
            cluster_vectors = [vectors[index][1] for index in cluster.get_indices()]
            new_centroid = [sum(coord[i] for i in range(len(coord))) / len(cluster_vectors)
                            for coord in cluster_vectors]
            cluster.set_new_centroid(tuple(new_centroid))

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
        if (not isinstance(query_vector, tuple) or
                not query_vector or
                not isinstance(n_neighbours, int) or
                not n_neighbours or
                n_neighbours <= 0):
            raise ValueError(f'Incorrect value: "{query_vector}" is not a tuple, empty, '
                             f'is None or is zero or '
                             f'"{n_neighbours}" is not an integer, empty, '
                             f'is None, is zero or is less than zero.')

        centroid_distances = []
        for cluster_index, cluster in enumerate(self.__clusters):
            centroid = cluster.get_centroid()
            if centroid is None:
                continue

            centroid_distance = calculate_distance(query_vector, cluster.get_centroid())
            if centroid_distance is None:
                raise ValueError(f'Incorrect value: "{centroid_distance}" is empty, '
                                 f'is None or is zero.')

            centroid_distances.append((centroid_distance, cluster_index))

        min_centroid_index = centroid_distances.index(min(centroid_distances))
        cluster_indices = self.__clusters[min_centroid_index].get_indices()
        cluster_vectors = self._db.get_vectors(cluster_indices)

        vector_distances = []
        for vector_index, vector in cluster_vectors:
            distance = calculate_distance(query_vector, vector)
            if distance is None:
                raise ValueError(f'Incorrect value: "{distance}" is empty, '
                                 f'is None or is zero.')
            vector_distances.append((distance, vector_index))

        return sorted(vector_distances, key=lambda x: x[0])[:n_neighbours]

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
        if (not isinstance(new_clusters, list) or
                not new_clusters
                or not isinstance(threshold, float) or
                not threshold):
            raise ValueError(f'Incorrect value: "{new_clusters}" is not a list, is empty, '
                             f'is None or is zero or "{threshold}" is not a float, '
                             f'is empty, is None or is zero.')

        for i, old_cluster in enumerate(self.__clusters):
            distance = calculate_distance(old_cluster.get_centroid(),
                                          new_clusters[i].get_centroid())
            if not isinstance(distance, float):
                raise ValueError(f'Incorrect value: "{distance}" is not a float.')
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
        self._db = db
        self.__algo = KMeans(db, n_clusters)
        self.__algo.train()

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
        if (not query or
                not isinstance(query, str) or
                not n_neighbours or
                not isinstance(n_neighbours, int) or
                n_neighbours <= 0):
            raise ValueError(f'Incorrect value: "{query}" is not a string or empty or '
                             f'"{n_neighbours}" is not an integer, empty, '
                             f'is None, is zero or is less than zero.')

        query_token = self._db.get_tokenizer().tokenize(query)
        if query_token is None:
            raise ValueError(f'Incorrect value: "{query_token}" is empty, '
                             f'is None or is zero.')

        query_vector = self._db.get_vectorizer().vectorize(query_token)
        if query_vector is None:
            raise ValueError(f'Incorrect value: "{query_vector}" is empty, '
                             f'is None or is zero.')

        neighbours = self.__algo.infer(query_vector, n_neighbours)
        if not neighbours:
            raise ValueError(f'Incorrect value: "{neighbours}" is empty, '
                             f'is None or is zero.')

        document_indices = tuple(neighbour[-1] for neighbour in neighbours)
        raw_documents = self._db.get_raw_documents(document_indices)
        if not raw_documents:
            raise ValueError(f'Incorrect value: "{raw_documents}" is empty, '
                             f'is None or is zero.')

        relevant_documents = []
        for i, distance in enumerate(neighbours):
            relevant_documents.append((distance[0], raw_documents[i]))
        return relevant_documents

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
        self._db = db
        self._engine = engine

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
        return self._engine.retrieve_relevant_documents(query, n_neighbours=n_neighbours)


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
        super().__init__(db, SearchEngine(db.get_vectorizer(), db.get_tokenizer()))
        self._engine.index_documents(db.get_raw_documents())


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
        super().__init__(db, AdvancedSearchEngine(db.get_vectorizer(), db.get_tokenizer()))
        self._engine.index_documents(db.get_raw_documents())
