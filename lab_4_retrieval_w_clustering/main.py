"""
Lab 4.

Vector search with clusterization
"""
from math import sqrt
from lab_2_retrieval_w_bm25.main import calculate_bm25
from json import dump
# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called
from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vector, Vectorizer, calculate_distance, SearchEngine, AdvancedSearchEngine

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
        if not isinstance(tokenized_corpus, list) or not tokenized_corpus:
            raise ValueError
        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(map(len, self._corpus)) / len(self._corpus)

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
        if not isinstance(tokenized_document, list):
            raise ValueError
        vector = self._calculate_bm25(tokenized_document)
        if not vector:
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
        if not isinstance(tokenized_document, list):
            raise ValueError
        vector_list = [0.0 * n for n in range(len(self._vocabulary))]
        bm_dict = calculate_bm25(self._vocabulary, tokenized_document, self._idf_values, 1.5, 0.75, self._avg_doc_len,
                                 len(tokenized_document))
        if not bm_dict:
            return tuple(vector_list)
        for ind, word in enumerate(self._vocabulary):
            vector_list[ind] = bm_dict[word]
        return tuple(vector_list)


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
        self.__vectors = {}
        self.__documents = []

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
            raise ValueError
        self.__documents = corpus
        tokenized_docs = [tok_doc for doc in self.__documents if (tok_doc := self._tokenizer.tokenize(doc))]
        if not tokenized_docs:
            raise ValueError
        self._vectorizer.set_tokenized_corpus(tokenized_docs)
        self._vectorizer.build()
        for doc in tokenized_docs:
            vectorized = self._vectorizer.vectorize(doc)
            self.__vectors[tokenized_docs.index(doc)] = vectorized

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
        if not indices:
            return list(self.__vectors.items())
        unique = []
        for num in indices:
            if num not in unique:
                unique.append(num)
        return [(num, self.__vectors[num]) for num in unique]

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
        if not indices:
            return self.__documents
        unique = []
        for num in indices:
            if num not in unique:
                unique.append(num)
        return [self.__documents[num] for num in unique]


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
        if not isinstance(query, str) or not query or not isinstance(n_neighbours, int) or n_neighbours <= 0:
            raise ValueError
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        vectorized_query = self._db.get_vectorizer().vectorize(tokenized_query)
        vectors = [tup[1] for tup in self._db.get_vectors()]
        neighbours = self._calculate_knn(vectorized_query, vectors, n_neighbours)
        if not neighbours:
            raise ValueError
        documents = self._db.get_raw_documents(tuple([tup[0] for tup in neighbours]))
        return [(tup[-1], documents[ind]) for ind, tup in enumerate(neighbours)]


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
        if not isinstance(index, int) or index is None or index < 0:
            raise ValueError
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
        centroids = self._db.get_vectors()[:self._n_clusters]
        for centroid in centroids:
            self.__clusters.append(ClusterDTO(centroid[-1]))
        while True:
            new_clusters = self.run_single_train_iteration()
            if self._is_convergence_reached(new_clusters):
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
            for centroid in centroids:
                distances.append((calculate_distance(vector[-1], centroid), centroids.index(centroid)))
            closest = min(distances)
            self.__clusters[closest[-1]].add_document_index(vectors.index(vector))
        for cluster in self.__clusters:
            cluster_vectors = []
            for num in cluster.get_indices():
                cluster_vectors.append(vectors[num][-1])
            new_centroid = tuple(sum(value) / len(value) for value in zip(*cluster_vectors))
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
        if (not isinstance(query_vector, tuple) or not query_vector
                or not isinstance(n_neighbours, int) or not n_neighbours):
            raise ValueError
        distances = [(calculate_distance(query_vector, cluster.get_centroid()), ind)
                     for ind, cluster in enumerate(self.__clusters)]
        closest_cluster = min(distances)[-1]
        cluster_indices = self.__clusters[closest_cluster].get_indices()
        vectors = [self._db.get_vectors()[num] for num in cluster_indices]
        distances = []
        for index, vector in enumerate(vectors):
            distance = calculate_distance(query_vector, vector[-1])
            if distance is None:
                raise ValueError
            distances.append((distance, vector[0]))
        return sorted(distances, key=lambda x: x[-1])[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int| list[str]]]: List with information about each cluster
        """
        clusters_info = {}
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            cluster_indices = cluster.get_indices()
            vectors = [self._db.get_vectors()[num] for num in cluster_indices]
            distances = []
            for index, vector in enumerate(vectors):
                distance = calculate_distance(centroid, vector[-1])
                distances.append((distance, vector[0]))
            distances = sorted(distances, key=lambda x: x[-1])[:num_examples]
            indices = tuple(tup[-1] for tup in distances)
            docs = self._db.get_raw_documents(indices)
            clusters_info[f'{self.__clusters.index(cluster)}'] = docs
        return clusters_info

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        clusters_sum = 0.0
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            cluster_indices = cluster.get_indices()
            vectors = [self._db.get_vectors()[num] for num in cluster_indices]
            sum_ = 0.0
            for vector in vectors:
                sum_ += sqrt(sum((cx - vx) ** 2 for cx, vx in zip(centroid, vector[-1])))
            clusters_sum += sum_
        return clusters_sum

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
        if (not isinstance(new_clusters, list) or not new_clusters
                or not isinstance(threshold, float) or not threshold):
            raise ValueError
        for ind, cluster in enumerate(self.__clusters):
            old_centroid = cluster.get_centroid()
            new_centroid = new_clusters[ind].get_centroid()
            if not old_centroid or not new_centroid:
                raise ValueError
            distance = calculate_distance(old_centroid, new_centroid)
            if distance is None:
                raise ValueError
            if not distance < threshold:
                return False
            continue
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
        if not db or not isinstance(n_clusters, int):
            raise ValueError
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
        if (not isinstance(query, str) or not query
                or not isinstance(n_neighbours, int) or not n_neighbours):
            raise ValueError
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if not tokenized_query:
            raise ValueError
        query_vector = self._db.get_vectorizer().vectorize(tokenized_query)
        if not query_vector:
            raise ValueError
        self.__algo.train()
        relevant_distances = self.__algo.infer(query_vector, n_neighbours)
        if not relevant_distances:
            raise ValueError
        indices = tuple(tup[-1] for tup in relevant_distances)
        docs = self._db.get_raw_documents(indices)
        return [(neighbour[0], docs[ind]) for ind, neighbour in enumerate(relevant_distances)]

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        info = self.__algo.get_clusters_info(num_examples)
        with open(output_path, 'w', encoding='utf-8') as file_to_save:
            dump(info, file_to_save)

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

