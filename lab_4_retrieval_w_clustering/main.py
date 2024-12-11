"""
Lab 4.

Vector search with clusterization
"""
from lab_2_retrieval_w_bm25.main import calculate_bm25

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument,
# duplicate-code, unused-private-member, super-init-not-called
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
    if not isinstance(text, str) or not text:
        raise ValueError('A text must be a non-empty string. ')
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
        if (not isinstance(self._corpus, list) or tokenized_corpus == []
                or not all(isinstance(doc, list) for doc in self._corpus)):
            raise ValueError

        self._corpus = tokenized_corpus
        self._avg_doc_len = (sum(len(paragraph) for paragraph in tokenized_corpus)
                             / len(tokenized_corpus))

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
        if (not isinstance(tokenized_document, list) or not tokenized_document or
                not all(isinstance(token, str) for token in tokenized_document)):
            raise ValueError
        vectorized_document = self._calculate_bm25(tokenized_document)
        if vectorized_document is None:
            raise ValueError
        return vectorized_document

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
        if (not isinstance(tokenized_document, list) or not tokenized_document or
                not all(isinstance(token, str) for token in tokenized_document)):
            raise ValueError('Inappropriate type input argument / argument')

        vector = list(0.0 for _ in range(len(self._vocabulary)))
        bm25_values = calculate_bm25(self._vocabulary, tokenized_document, self._idf_values,
                                     1.5, 0.75, self._avg_doc_len, len(tokenized_document))
        for index, word in enumerate(self._vocabulary):
            if bm25_values is not None:
                vector[index] = bm25_values[word]

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
        if not corpus or not isinstance(corpus, list):
            raise ValueError("Corpus must be a non-empty list.")

        tokenized_texts = []
        for text in corpus:
            tokenized_text = self._tokenizer.tokenize(text)
            if tokenized_text:
                tokenized_texts.append(tokenized_text)
                self.__documents.append(text)
        if not tokenized_texts:
            raise ValueError("No valid documents after tokenization.")

        self._vectorizer.set_tokenized_corpus(tokenized_texts)
        self._vectorizer.build()

        for index, text in enumerate(tokenized_texts):
            self.__vectors[index] = self._vectorizer.vectorize(text)

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
        unique_indices = set(indices)
        return [(ind, self.__vectors[ind]) for ind in unique_indices]

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
        if not isinstance(indices, tuple):
            raise ValueError("Indices must be a tuple of integers.")

        unique_documents = []
        for index in indices:
            if self.__documents[index] not in unique_documents:
                unique_documents.append(self.__documents[index])
        return unique_documents


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
        if (not query or not isinstance(query, str)
                or n_neighbours <= 0 or not isinstance(n_neighbours, int)):
            raise ValueError('Query must be a non-empty string; '
                             'Number of neighbours must be a positive integer.')

        tokenized_query = self._db.get_tokenizer().tokenize(query)
        vectorized_query = self._db.get_vectorizer().vectorize(tokenized_query)

        vectors = [vector[1] for vector in self._db.get_vectors()]
        neighbours = self._calculate_knn(vectorized_query, vectors, n_neighbours)
        if not neighbours:
            raise ValueError('No neighbours found.')

        relevant_documents = self._db.get_raw_documents(tuple(neighbour[0]
                                                              for neighbour in neighbours))
        return [(distance[-1], relevant_documents[ind])
                for ind, distance in enumerate(neighbours)]


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
        if not isinstance(new_centroid, tuple) or not new_centroid:
            raise ValueError('New centroid must be a non-empty tuple.')
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
        if index is None or not isinstance(index, int) or index < 0:
            raise ValueError('Index must be a non-empty positive integer. ')

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
        self.__clusters = []
        self._db = db
        self._n_clusters = n_clusters

    def train(self) -> None:
        """
        Train k-means algorithm.
        """
        new_centroids = self._db.get_vectors()[:self._n_clusters]
        self.__clusters = [ClusterDTO(centroid[1]) for centroid in new_centroids]
        self.run_single_train_iteration()
        while self._is_convergence_reached(self.__clusters) is False:
            self.run_single_train_iteration()
            self._is_convergence_reached(self.__clusters)

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
            centroid = cluster.get_centroid()
            centroids.append(centroid)
        vectors = self._db.get_vectors()
        for vector in vectors:
            distances = []
            for centroid in centroids:
                distance = calculate_distance(vector[1], centroid)
                if distance is None:
                    raise ValueError('Distance must not be NoneType. ')
                distances.append((distance, centroids.index(centroid)))
            closest_ind = min(distances)[1]
            self.__clusters[closest_ind].add_document_index(vectors.index(vector))
        for cluster in self.__clusters:
            cluster_vectors = []
            for ind in cluster.get_indices():
                cluster_vectors.append(vectors[ind][1])
            new_centroid = tuple([sum(vector[i] for i in range(len(vector))) /
                                  len(cluster_vectors) for vector in cluster_vectors])
            cluster.set_new_centroid(new_centroid)
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
        if (not query_vector or not isinstance(query_vector, tuple)
                or not n_neighbours or not isinstance(n_neighbours, int)):
            raise ValueError('Query vector must be a non-empty tuple. '
                             'Number of neighbours must be a non-empty integer. ')
        centroid_distances = []
        for cluster in self.__clusters:
            distance = calculate_distance(query_vector, cluster.get_centroid())
            if distance is None:
                raise ValueError('Distance must not be NoneType. ')
            centroid_distances.append(distance)
        closest_cluster = self.__clusters[centroid_distances.index(min(centroid_distances))]
        if not closest_cluster.get_centroid():
            closest_cluster = self.__clusters[0]
        indices = closest_cluster.get_indices()
        if indices is None:
            raise ValueError('Indices must not be NoneType. ')
        cluster_vectors = self._db.get_vectors(indices)
        if cluster_vectors is None:
            raise ValueError('Cluster vectors must not be NoneType. ')
        vector_distances = []
        for vector in cluster_vectors:
            distance = calculate_distance(query_vector, vector[-1])
            if distance is None:
                raise ValueError('Distance must not be NoneType. ')
            vector_distances.append((distance, vector[0]))

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
        if not new_clusters:
            raise ValueError('New clusters must be non-empty. ')

        for ind, old_cluster in enumerate(self.__clusters):
            distance = calculate_distance(old_cluster.get_centroid(),
                                          new_clusters[ind].get_centroid())
            if distance is None:
                raise ValueError('Distance must not be NoneType. ')
            if threshold <= distance:
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
        if (not query or not isinstance(query, str) or not n_neighbours
                or not isinstance(n_neighbours, int) or n_neighbours <= 0):
            raise ValueError
        query_token = self._db.get_tokenizer().tokenize(query)
        if query_token is None:
            raise ValueError
        query_vector = self._db.get_vectorizer().vectorize(query_token)
        if query_vector is None:
            raise ValueError
        neighbours = self.__algo.infer(query_vector, n_neighbours)
        if not neighbours:
            raise ValueError
        document_indices = tuple(neighbour[-1] for neighbour in neighbours)
        relevant_documents = self._db.get_raw_documents(document_indices)
        if not relevant_documents:
            raise ValueError
        return [(distance[0], relevant_documents[ind]) for ind, distance in enumerate(neighbours)]

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
