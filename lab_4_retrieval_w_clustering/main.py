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
    if not text or not isinstance(text, str):
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
        self._corpus = []
        self._avg_doc_len = -1.0
        super().__init__(self._corpus)

    def set_tokenized_corpus(self, tokenized_corpus: TokenizedCorpus) -> None:
        """
        Set tokenized corpus and average document length.

        Args:
            tokenized_corpus (TokenizedCorpus): Tokenized texts corpus.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not tokenized_corpus or not isinstance(tokenized_corpus, list) \
                or not all(isinstance(elem, list) for elem in tokenized_corpus):
            raise ValueError
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
        if not tokenized_document or not isinstance(tokenized_document, list) \
                or not all(isinstance(elem, str) for elem in tokenized_document):
            raise ValueError
        vector = self._calculate_bm25(tokenized_document)
        if vector is None:
            raise ValueError
        return vector

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
        if not tokenized_document or not isinstance(tokenized_document, list) \
                or not all(isinstance(elem, str) for elem in tokenized_document):
            raise ValueError()
        bm25_vector = [0.0] * len(self._vocabulary)
        bm25_results = calculate_bm25(self._vocabulary, tokenized_document,
                                      self._idf_values, avg_doc_len=self._avg_doc_len,
                                      doc_len=len(tokenized_document))
        if bm25_results is None or not bm25_results:
            return ()
        for word, index in self._token2ind.items():
            bm25_vector[index] = bm25_results[word]
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
        self.__documents = Corpus()
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
        if not corpus or not isinstance(corpus, list) \
                or not all(isinstance(elem, str) for elem in corpus):
            raise ValueError
        tokenized_documents = []
        for doc in corpus:
            tokenized_doc = self._tokenizer.tokenize(doc)
            if tokenized_doc:
                tokenized_documents.append(tokenized_doc)
                self.__documents.append(doc)
        self._vectorizer.set_tokenized_corpus(tokenized_documents)
        self._vectorizer.build()
        vectors = []
        for tokenized_doc in tokenized_documents:
            vectors.append(self._vectorizer.vectorize(tokenized_doc))
        self.__vectors = dict(enumerate(vectors))

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
        pairs = []
        for index in indices:
            pairs.append((index, self.__vectors[index]))
        return pairs

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
        corpus = []
        for index in indices:
            if self.__documents[index] not in corpus:
                corpus.append(self.__documents[index])
        return corpus


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
        if not isinstance(query, str) or not isinstance(n_neighbours, int) or n_neighbours <= 0:
            raise ValueError('An inappropriate type input arguments or input arguments are empty')
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if tokenized_query is None:
            raise ValueError('The method returned None')
        vector_query = self._db.get_vectorizer().vectorize(tokenized_query)
        if vector_query is None:
            raise ValueError('The method returned None')
        vectors = [pair[1] for pair in self._db.get_vectors()]
        most_relevant = self._calculate_knn(vector_query, vectors, n_neighbours)
        if most_relevant is None:
            raise ValueError
        indices = tuple(pair[0] for pair in most_relevant)
        raw_documents = self._db.get_raw_documents(indices)
        return [(most_relevant[num][1], raw_documents[num]) for num, index in enumerate(indices)]


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
        if not new_centroid or not isinstance(new_centroid, tuple):
            raise ValueError('An inappropriate type input arguments or input arguments are empty')
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
            raise ValueError('An inappropriate type input arguments or input arguments are empty')
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
        initial_centroids = self._db.get_vectors()[:self._n_clusters]
        self.__clusters = [ClusterDTO(centroid[1]) for centroid in initial_centroids]
        self.run_single_train_iteration()
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
            centroid = cluster.get_centroid()
            centroids.append(centroid)
        vectors = self._db.get_vectors()
        for vector in vectors:
            distances = []
            for current_centroid in centroids:
                distance = calculate_distance(current_centroid, vector[1])
                if distance is None:
                    raise ValueError('The function returned None')
                distances.append(distance)
            closest = min(distances)
            self.__clusters[distances.index(closest)].add_document_index(vector[0])
        for current_cluster in self.__clusters:
            docs_vectors = [vector[1] for index in current_cluster.get_indices() for vector
                            in vectors if index == vector[0]]
            new_centroid = tuple(sum(row) / len(docs_vectors) for row in zip(*docs_vectors))
            current_cluster.set_new_centroid(new_centroid)
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
        if not query_vector or not isinstance(query_vector, tuple) \
                or not n_neighbours or not isinstance(n_neighbours, int):
            raise ValueError('An inappropriate type input arguments or input arguments are empty')
        query2centroid = []
        for cluster in self.__clusters:
            distance = calculate_distance(query_vector, cluster.get_centroid())
            if distance is None:
                raise ValueError('The function returned None')
            query2centroid.append(distance)
        nearest = sorted(query2centroid)[0]
        nearest_cluster = self.__clusters[query2centroid.index(nearest)]
        if not nearest_cluster.get_centroid():
            nearest_cluster = self.__clusters[0]
        indices = nearest_cluster.get_indices()
        docs_vectors = self._db.get_vectors(indices)
        query2vector = []
        for vector in docs_vectors:
            distance = calculate_distance(query_vector, vector[1])
            if distance is None:
                raise ValueError('The function returned None')
            query2vector.append((distance, vector[0]))
        return sorted(query2vector, key=lambda x: float(x[0]))[:n_neighbours]

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
        if not new_clusters or not isinstance(new_clusters, list) \
                or not isinstance(threshold, float):
            raise ValueError('An inappropriate type input arguments or input arguments are empty')
        old_centroids = [cluster.get_centroid() for cluster in self.__clusters]
        for index, centroid in enumerate(new_clusters):
            new_centroid = centroid.get_centroid()
            distance = calculate_distance(new_centroid, old_centroids[index])
            if distance is None:
                raise ValueError('The function returned None')
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
        if not query or not isinstance(query, str) or not isinstance(n_neighbours, int) \
                or n_neighbours <= 0:
            raise ValueError('An inappropriate type input arguments or input arguments are empty')
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if tokenized_query is None:
            raise ValueError('The method returned None')
        vector_query = self._db.get_vectorizer().vectorize(tokenized_query)
        if vector_query is None:
            raise ValueError('The method returned None')
        self.__algo.train()
        most_relevant = self.__algo.infer(vector_query, n_neighbours)
        if most_relevant is None:
            raise ValueError
        indices = tuple(pair[1] for pair in most_relevant)
        raw_documents = self._db.get_raw_documents(indices)
        return [(most_relevant[num][0], raw_documents[num]) for num, index in enumerate(indices)]

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
