"""
Lab 4.

Vector search with clusterization
"""
import json

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code,
# unused-private-member, super-init-not-called

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
    if not text or not isinstance(text, str):
        raise ValueError('Invalid input')
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
        self._avg_doc_len = -1

    def set_tokenized_corpus(self, tokenized_corpus: TokenizedCorpus) -> None:
        """
        Set tokenized corpus and average document length.

        Args:
            tokenized_corpus (TokenizedCorpus): Tokenized texts corpus.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not tokenized_corpus:
            raise ValueError('Invalid input')
        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(len(doc) for doc in self._corpus) / len(self._corpus)

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
        if not (tokenized_document and isinstance(tokenized_document, list)
                and all(isinstance(doc, str) for doc in tokenized_document)):
            raise ValueError('Invalid input')
        vector_bm25 = self._calculate_bm25(tokenized_document)
        if vector_bm25 is None:
            raise ValueError('Function returned None')
        return vector_bm25

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
        if not (tokenized_document and isinstance(tokenized_document, list)
                and all(isinstance(doc, str) for doc in tokenized_document)):
            raise ValueError('Invalid input')
        bm25_vector = [0.0] * len(self._vocabulary)
        idf = calculate_idf(self._vocabulary, self._corpus)
        if not idf:
            return Vector(bm25_vector)
        bm25 = calculate_bm25(self._vocabulary, tokenized_document, idf,
                              avg_doc_len=self._avg_doc_len, doc_len=len(tokenized_document))
        if not bm25:
            return Vector(bm25_vector)
        for token in bm25:
            if token in self._vocabulary:
                bm25_vector[self._token2ind[token]] = bm25[token]
        return Vector(bm25_vector)


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
            raise ValueError('Invalid input')

        tokenized_corpus: list = []
        for doc in corpus:
            tokenized_doc = self._tokenizer.tokenize(doc)
            if tokenized_doc:
                self.__documents.append(doc)
                tokenized_corpus.append(tokenized_doc)

        self._vectorizer.set_tokenized_corpus(tokenized_corpus)
        self._vectorizer.build()
        for idx, doc in enumerate(tokenized_corpus):
            self.__vectors[idx] = self._vectorizer.vectorize(list(doc))

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
        return [(idx, self.__vectors[idx]) for idx in indices]

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
        if not (isinstance(indices, tuple) and all(isinstance(index, int) for index in indices)):
            raise ValueError('Invalid input')
        return [self.__documents[idx] for idx in set(indices)]


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
        if not (query and isinstance(query, str) and isinstance(n_neighbours,
                                                                int) and n_neighbours > 0):
            raise ValueError('Invalid input')

        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if tokenized_query is None:
            raise ValueError('Function returned None')

        vector_query = self._db.get_vectorizer().vectorize(tokenized_query)
        if vector_query is None:
            raise ValueError('Function returned None')

        vector_documents = [doc[1] for doc in self._db.get_vectors()]

        relevant_documents = self._calculate_knn(vector_query, vector_documents, n_neighbours)
        if relevant_documents is None:
            raise ValueError('Function returned None')

        return [(doc[1], self._db.get_raw_documents(tuple([doc[0]]))[0])
                for doc in relevant_documents]


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
        if not (new_centroid and isinstance(new_centroid, tuple) and
                all(isinstance(cor, float) for cor in new_centroid)):
            raise ValueError('Invalid input')
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
        if not (isinstance(index, int) and index >= 0):
            raise ValueError('Invalid input')
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
        initial_centroids = [pair[1] for pair in self._db.get_vectors()[:self._n_clusters]]
        for centroid in initial_centroids:
            self.__clusters.append(ClusterDTO(centroid))
        new_clusters = self.run_single_train_iteration()
        while not self._is_convergence_reached(new_clusters):
            new_clusters = self.run_single_train_iteration()
        new_clusters = self.run_single_train_iteration()
        self.__clusters = new_clusters

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        clusters = self.__clusters.copy()
        for cluster in clusters:
            cluster.erase_indices()
        for vector_doc in self._db.get_vectors():
            nearest_cluster = clusters[0]
            for cluster in clusters:
                centroid = cluster.get_centroid()
                distance = calculate_distance(vector_doc[1], centroid)
                distance_to_nearest = calculate_distance(vector_doc[1],
                                                         nearest_cluster.get_centroid())
                if distance is None or distance_to_nearest is None:
                    raise ValueError('Function returned None')
                if distance < distance_to_nearest:
                    nearest_cluster = cluster
            nearest_cluster.add_document_index(vector_doc[0])
        for cluster in clusters:
            cluster_vectors_with_indices = self._db.get_vectors(cluster.get_indices())
            if not cluster_vectors_with_indices:
                cluster.set_new_centroid(())
                continue
            cluster_vectors = [pair[1] for pair in cluster_vectors_with_indices]
            new_centroid = [0.0] * len(cluster_vectors[0])
            for idx, _ in enumerate(new_centroid):
                new_centroid[idx] = sum(tup[idx] for tup in cluster_vectors) / len(cluster_vectors)
            cluster.set_new_centroid(Vector(new_centroid))
        return clusters

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
        if not (n_neighbours and isinstance(n_neighbours, int) and query_vector
                and isinstance(query_vector, tuple) and all(isinstance(cor, float)
                                                            for cor in query_vector)):
            raise ValueError('Invalid input')
        closest_cluster = self.__clusters[0]

        for cluster in self.__clusters:
            dist_cluster = calculate_distance(query_vector, cluster.get_centroid())
            dist_closest_cluster = calculate_distance(query_vector, closest_cluster.get_centroid())
            if dist_cluster is None or dist_closest_cluster is None:
                raise ValueError('Function returned None')
            if dist_cluster < dist_closest_cluster:
                closest_cluster = cluster

        indices = closest_cluster.get_indices()
        docs_vectors = self._db.get_vectors(indices)
        docs_with_distance = []
        for doc in docs_vectors:
            distance = calculate_distance(query_vector, doc[1])
            if distance is None:
                raise ValueError('Function returned None')
            docs_with_distance.append((distance, doc[0]))
        docs_with_distance.sort(key=lambda x: x[0])

        return docs_with_distance[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int| list[str]]]: List with information about each cluster
        """
        if not (num_examples and isinstance(num_examples, int) and num_examples > 0):
            raise ValueError('Invalid input')
        info: list[dict] = []
        for cluster in self.__clusters:
            cluster_info = []
            centroid = cluster.get_centroid()
            documents_indices = cluster.get_indices()
            docs_vectors_with_indices = self._db.get_vectors(documents_indices)
            docs_vectors = [pair[1] for pair in docs_vectors_with_indices]
            for idx, docs_vector in enumerate(docs_vectors):
                distance = calculate_distance(centroid, docs_vector)
                if distance is None:
                    raise ValueError('Function returned None')
                cluster_info.append((distance,
                                     *self._db.get_raw_documents((documents_indices[idx],))))
            cluster_info.sort(key=lambda x: bool(x[0]))
            info.append({
                'cluster_id': self.__clusters.index(cluster),
                'documents': [pair[1] for pair in cluster_info][:num_examples]
            })
        return info

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        result = 0.0
        for cluster in self.__clusters:
            cluster_result = 0.0
            centroid = cluster.get_centroid()
            documents_indices = cluster.get_indices()
            docs_vectors_with_indices = self._db.get_vectors(documents_indices)
            docs_vectors = [pair[1] for pair in docs_vectors_with_indices]
            for doc_vector in docs_vectors:
                cluster_result += sum(centroid[idx] ** 2 - doc_vector[idx] ** 2
                                       for idx in range(len(centroid)))
            result += cluster_result
        return result

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
        if not (new_clusters and threshold and isinstance(new_clusters, list)
                and isinstance(threshold, float)):
            raise ValueError('Invalid input')
        for idx, new_cluster in enumerate(new_clusters):
            distance = calculate_distance(self.__clusters[idx].get_centroid(),
                                          new_cluster.get_centroid())
            if distance is None:
                raise ValueError('Invalid input')
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
        self.__algo = KMeans(self._db, n_clusters)
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
        if not (query and isinstance(query, str) and
                isinstance(n_neighbours, int) and n_neighbours > 0):
            raise ValueError('Invalid input')
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if tokenized_query is None:
            raise ValueError('Function returned None')
        vectorized_query = self._db.get_vectorizer().vectorize(tokenized_query)
        if vectorized_query is None:
            raise ValueError('Function returned None')
        relevant_docs_indices_with_distance = self.__algo.infer(vectorized_query, n_neighbours)
        if relevant_docs_indices_with_distance is None:
            raise ValueError('Function returned None')
        indices = tuple(pair[1] for pair in relevant_docs_indices_with_distance)
        relevant_docs = self._db.get_raw_documents(indices)
        relevant_docs_with_distance = []
        for idx in range(len(indices)):
            relevant_docs_with_distance.append((relevant_docs_indices_with_distance[idx][0],
                                                relevant_docs[idx]))
        return relevant_docs_with_distance

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        with open(output_path, "w", encoding="UTF-8") as f:
            json.dump(self.__algo.get_clusters_info(num_examples), f)

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
