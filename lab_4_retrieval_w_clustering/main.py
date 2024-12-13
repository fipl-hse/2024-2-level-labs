"""
Lab 4.

Vector search with clusterization
"""
import json

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

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument,
# duplicate-code, unused-private-member, super-init-not-called


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
        raise ValueError('Invalid input: text must be a non-empty string.')
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
            raise ValueError('Invalid input: self._corpus must be a list, '
                             'tokenized_corpus must be a non-empty list of lists.')

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
            raise ValueError('Invalid input: tokenized_document must be '
                             'a non-empty list of strings.')
        vectorized_document = self._calculate_bm25(tokenized_document)
        if vectorized_document is None:
            raise ValueError('Method _calculate_bm25() returns None!')
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
            raise ValueError('Invalid input: tokenized_document must be '
                             'a non-empty list of strings.')

        vector = [0.0] * len(self._vocabulary)
        bm25_values = calculate_bm25(self._vocabulary, tokenized_document,
                                     self._idf_values, avg_doc_len=self._avg_doc_len,
                                     doc_len=len(tokenized_document))
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
            raise ValueError('Invalid input: corpus must be a non-empty list.')

        tokenized_texts = []
        for text in corpus:
            tokenized_text = self._tokenizer.tokenize(text)
            if tokenized_text:
                tokenized_texts.append(tokenized_text)
                self.__documents.append(text)
        if not tokenized_texts:
            raise ValueError('No valid documents after tokenization!')

        self._vectorizer.set_tokenized_corpus(tokenized_texts)
        self._vectorizer.build()

        for index, tokenized_text in enumerate(tokenized_texts):
            self.__vectors[index] = self._vectorizer.vectorize(tokenized_text)

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
            raise ValueError('Invalid input: indices must be a tuple of integers or None.')

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
                or n_neighbours < 1 or not isinstance(n_neighbours, int)):
            raise ValueError('Invalid input: query must be a non-empty string;'
                             'n_neighbours must be a positive integer.')

        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if tokenized_query is None:
            raise ValueError('Method get_tokenizer() returns None!')
        vectorized_query = self._db.get_vectorizer().vectorize(tokenized_query)
        vectors = [vector[1] for vector in self._db.get_vectors()]
        neighbours = self._calculate_knn(vectorized_query, vectors, n_neighbours)
        if not neighbours:
            raise ValueError('Method _calculate_knn() returns None!')

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
            raise ValueError('Invalid input: new_centroid must be a non-empty tuple.')
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
            raise ValueError('Invalid input: index must be a non-empty positive integer.')

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
        while not self._is_convergence_reached(self.__clusters):
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
                    raise ValueError('Method calculate_distance() returns None!')
                distances.append((distance, centroids.index(centroid)))
            closest_ind = min(distances)[1]
            self.__clusters[closest_ind].add_document_index(vectors.index(vector))

        for cluster in self.__clusters:
            vectors = [v for i, v in self._db.get_vectors(cluster.get_indices())]
            new_centroid = [sum(x)/len(x) for x in zip(*vectors)]
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
        if (not query_vector or not isinstance(query_vector, tuple)
                or not n_neighbours or not isinstance(n_neighbours, int)):
            raise ValueError('Invalid input: query vector must be a non-empty tuple, '
                             'n_neighbours must be a non-empty integer.')
        centroid_distances = []
        for cluster in self.__clusters:
            if cluster.get_centroid() is None:
                continue
            distance = calculate_distance(query_vector, cluster.get_centroid())
            if distance is None:
                raise ValueError('Method calculate_distance() returns None!')
            centroid_distances.append(distance)
        closest_cluster = self.__clusters[centroid_distances.index(min(centroid_distances))]
        if not closest_cluster.get_centroid():
            closest_cluster = self.__clusters[0]
        indices = closest_cluster.get_indices()
        if indices is None:
            raise ValueError('Method get_indices() returns None!')
        cluster_vectors = self._db.get_vectors(indices)
        if cluster_vectors is None:
            raise ValueError('Method get_vectors() returns None!')
        vector_distances = []
        for vector in cluster_vectors:
            distance = calculate_distance(query_vector, vector[-1])
            if distance is None:
                raise ValueError('Method calculate_distance() returns None!')
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
        if not isinstance(num_examples, int) or num_examples <= 0:
            raise ValueError('Invalid input: num_examples must be a positive integer.')
        if not self.__clusters:
            return []
        clusters_inf = []
        for ind, cluster in enumerate(self.__clusters):
            cluster_indices = cluster.get_indices()
            if not cluster_indices:
                continue
            distances = []
            for index in cluster_indices:
                vector_data = self._db.get_vectors()[index]
                vector_ind, vector = vector_data[0], vector_data[-1]
                distance = calculate_distance(cluster.get_centroid(), vector)
                if distance is None:
                    raise ValueError('Method calculate_distance() returns None!')
                distances.append((distance, vector_ind))
            distances = sorted(distances, key=lambda x: x[0])[:num_examples]
            indices = [dist[1] for dist in distances]
            docs = self._db.get_raw_documents(tuple(indices))
            more_inf = {}
            if isinstance(ind, int) and isinstance(docs, list):
                more_inf.update(cluster_id=ind, documents=docs)
            clusters_inf.append(more_inf)
        return clusters_inf

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        sse_result = 0.0
        vectors = self._db.get_vectors()
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            cluster_indices = cluster.get_indices()
            cluster_vectors = [vectors[ind][1] for ind in cluster_indices]
            sse = sum(sum((centroid[i] - vector[i]) ** 2 for
                          i in range(len(centroid))) for vector in cluster_vectors)
            sse_result += sse
        return sse_result

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
            raise ValueError('Invalid input: new_clusters must be a non-empty list.')

        for old_cluster, new_cluster in zip(self.__clusters, new_clusters):
            distance = calculate_distance(old_cluster.get_centroid(),
                                          new_cluster.get_centroid())
            if distance is None:
                raise ValueError('Method calculate_distance() returns None!')
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
            raise ValueError('Invalid input: query must be a non-empty string, '
                             'n_neighbours must be a positive integer.')
        query_token = self._db.get_tokenizer().tokenize(query)
        if query_token is None:
            raise ValueError('Method tokenize() returns None!')
        query_vector = self._db.get_vectorizer().vectorize(query_token)
        if query_vector is None:
            raise ValueError('Method vectorize() returns None!')
        neighbours = self.__algo.infer(query_vector, n_neighbours)
        if not neighbours:
            raise ValueError('Method infer() returns None!')
        document_indices = tuple(neighbour[-1] for neighbour in neighbours)
        relevant_documents = self._db.get_raw_documents(document_indices)
        if not relevant_documents:
            raise ValueError('Method get_raw_documents() returns None!')
        return [(distance[0], relevant_documents[ind]) for ind, distance in enumerate(neighbours)]

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(self.__algo.get_clusters_info(num_examples), file)

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
