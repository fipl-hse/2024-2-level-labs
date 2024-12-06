"""
Lab 4.

Vector search with clusterization
"""

from json import dump

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called
from lab_2_retrieval_w_bm25.main import calculate_bm25
from lab_3_ann_retriever.main import (
    AdvancedSearchEngine,
    BasicSearchEngine,
    calculate_distance,
    SearchEngine,
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
        raise ValueError

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
        super().__init__([])
        self._corpus = []
        self._avg_doc_len = -1

    def set_tokenized_corpus(self, tokenized_corpus: TokenizedCorpus) -> None:
        """
        Set tokenized corpus and average document length.

        Args:
            tokenized_corpus (TokenizedCorpus): Tokenized texts corpus.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(tokenized_corpus, list) or not tokenized_corpus:
            raise ValueError

        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(len(paragraph) for paragraph in
                                tokenized_corpus) / len(tokenized_corpus)

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
        if not isinstance(tokenized_document, list) or not tokenized_document:
            raise ValueError

        result = self._calculate_bm25(tokenized_document)
        if not result:
            raise ValueError
        return result

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
        if not isinstance(tokenized_document, list) or not tokenized_document:
            raise ValueError

        bm25 = calculate_bm25(self._vocabulary, tokenized_document, self._idf_values,
                              avg_doc_len=self._avg_doc_len, doc_len=len(tokenized_document))
        bm25_vector = [0.0] * len(self._vocabulary)
        if isinstance(bm25, dict):
            for word, index in self._token2ind.items():
                bm25_vector[index] = bm25[word]
        return tuple(bm25_vector)


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
        if not isinstance(corpus, list) or not corpus:
            raise ValueError

        tokenized_corpus = []
        for text in corpus:
            tokenized_text = self._tokenizer.tokenize(text)
            if tokenized_text:
                tokenized_corpus.append(tokenized_text)
                self.__documents.append(text)
        if not tokenized_corpus:
            raise ValueError

        self._vectorizer.set_tokenized_corpus(tokenized_corpus)
        self._vectorizer.build()
        self.__vectors = {index: self._vectorizer.vectorize(tokenized_text)
                          for index, tokenized_text in enumerate(tokenized_corpus)}

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
        if not isinstance(indices, tuple):
            raise ValueError

        return [self.__documents[index] for index in set(indices)]


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
        if not isinstance(query, str) or not query or not isinstance(n_neighbours, int) \
                or n_neighbours <= 0:
            raise ValueError

        tokenizer, vectorizer = self._db.get_tokenizer(), self._db.get_vectorizer()
        tokenized_query = tokenizer.tokenize(query)
        if tokenized_query is None:
            raise ValueError
        vectorized_query = vectorizer.vectorize(tokenized_query)

        vectors = [vector[1] for vector in self._db.get_vectors()]
        neighbours = self._calculate_knn(vectorized_query, vectors, n_neighbours)
        if neighbours is None:
            raise ValueError
        documents = self._db.get_raw_documents(tuple(pair[0] for pair in neighbours))
        return [(distance, documents[index]) for index, distance in neighbours]


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
            raise ValueError
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
        if not isinstance(index, int) or index < 0:
            raise ValueError
        self.__indices.append(index)
        self.__indices = list(set(self.__indices))

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
        n_vectors = self._db.get_vectors()[:self._n_clusters]
        self.__clusters = [ClusterDTO(centroid[-1]) for centroid in n_vectors]
        while True:
            self.run_single_train_iteration()
            if self._is_convergence_reached(self.__clusters):
                break

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

        for index, vector in self._db.get_vectors():
            distances = []
            for centroid in centroids:
                distance = calculate_distance(vector, centroid)
                if distance is not None:
                    distances.append(distance)
                else:
                    raise ValueError

            nearest_centroid = min(distances)
            self.__clusters[distances.index(nearest_centroid)].add_document_index(index)

        vectors = self._db.get_vectors()
        for cluster in self.__clusters:
            cluster_vectors = [vectors[index][-1] for index in cluster.get_indices()]
            centroid_vector = [sum(vector[index] for index in range(
                len(vector))) / len(cluster_vectors) for vector in cluster_vectors]
            cluster.set_new_centroid(tuple(centroid_vector))
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
        if not isinstance(query_vector, tuple) or not query_vector or \
                not isinstance(n_neighbours, int) or n_neighbours <= 0:
            raise ValueError

        min_distance_cluster: tuple[float, ClusterDTO] = (100, self.__clusters[0])
        for cluster in self.__clusters:
            distance = calculate_distance(cluster.get_centroid(), query_vector)
            if distance is None:
                raise ValueError
            if distance < min_distance_cluster[0]:
                min_distance_cluster = (distance, cluster)

        indices = min_distance_cluster[1].get_indices()
        vectors = self._db.get_vectors(indices)

        distances = []
        for vector in vectors:
            distance = calculate_distance(vector[1], query_vector)
            if distance is None:
                raise ValueError
            distances.append((distance, vector[0]))

        return sorted(distances, key=lambda x: x[0])[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int| list[str]]]: List with information about each cluster
        """
        if not isinstance(num_examples, int) or num_examples <= 0:
            raise ValueError

        clusters_info: list[dict[str, int | list[str]]] = []
        for cluster in self.__clusters:
            inference = self.infer(cluster.get_centroid(), num_examples)
            indices = tuple(pair[1] for pair in inference)
            documents = self._db.get_raw_documents(indices)
            if documents is None:
                raise ValueError
            clusters_info.append({
                'cluster_id': self.__clusters.index(cluster),
                'documents': documents
            })
        return clusters_info

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        sse = 0.0
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            indices = cluster.get_indices()
            vectors = self._db.get_vectors(indices)
            for vector in vectors:
                for index, value in enumerate(vector[1]):
                    sse += (value - centroid[index]) ** 2
        return sse

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
        if not isinstance(new_clusters, list) or not new_clusters or \
                not isinstance(threshold, float):
            raise ValueError

        old_clusters = [cluster.get_centroid() for cluster in self.__clusters]
        for index, value in enumerate(new_clusters):
            distance = calculate_distance(value.get_centroid(), old_clusters[index])
            if distance is None:
                raise ValueError
            if distance >= threshold:
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
        if not isinstance(query, str) or not query or not isinstance(n_neighbours, int) or \
                n_neighbours <= 0:
            raise ValueError

        query_tokenized = self._db.get_tokenizer().tokenize(query)
        if query_tokenized is None:
            raise ValueError
        query_vectorized = self._db.get_vectorizer().vectorize(query_tokenized)
        if query_vectorized is None:
            raise ValueError

        infer_result = self.__algo.infer(query_vectorized, n_neighbours)
        indices = [neighbour[1] for neighbour in infer_result]
        distances = [neighbour[0] for neighbour in infer_result]
        raw_documents = self._db.get_raw_documents(tuple(indices))
        relevant_documents = list(zip(distances, raw_documents))
        return relevant_documents

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        if not isinstance(num_examples, int) or num_examples <= 0 or \
                not isinstance(output_path, str) or not output_path:
            raise ValueError

        with open(output_path, 'w', encoding='utf-8') as file:
            dump(self.__algo.get_clusters_info(num_examples), file)

    def calculate_square_sum(self) -> float:
        """
        Get sum by all clusters of sum of squares of distance from vector of clusters to centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        return self.__algo.calculate_square_sum()


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
        if not isinstance(query, str) or not query or \
                not isinstance(n_neighbours, int) or n_neighbours <= 0:
            raise ValueError
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
        self._engine.index_documents(self._db.get_raw_documents())
