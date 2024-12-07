"""
Lab 4.

Vector search with clusterization
"""

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code,
# unused-private-member, super-init-not-called
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
        if not isinstance(tokenized_corpus, list) or not tokenized_corpus\
                or not all(isinstance(paragraph, list) for paragraph in tokenized_corpus)\
                or not all(isinstance(token, str)
                           for paragraph in tokenized_corpus for token in paragraph):
            raise ValueError
        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(len(paragraph) for paragraph in self._corpus) / len(self._corpus)

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
        if not isinstance(tokenized_document, list) or not tokenized_document\
                or not all(isinstance(token, str) for token in tokenized_document):
            raise ValueError
        bm_25 = self._calculate_bm25(tokenized_document)
        if not bm_25:
            raise ValueError
        return bm_25

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
        if not isinstance(tokenized_document, list) or not tokenized_document\
                or not all(isinstance(token, str) for token in tokenized_document):
            raise ValueError
        vector = [0.0] * len(self._vocabulary)
        bm_25 = calculate_bm25(self._vocabulary, tokenized_document, self._idf_values, 1.5, 0.75,
                               self._avg_doc_len, len(tokenized_document))
        for index, token in enumerate(self._vocabulary):
            if isinstance(bm_25, dict):
                vector[index] = bm_25[token]
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
        self.__documents = []
        self.__vectors = {}
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
        if not isinstance(corpus, list) or not corpus:
            raise ValueError
        all_tokens = []
        for doc in corpus:
            tokens = self._tokenizer.tokenize(doc)
            if tokens:
                all_tokens.append(tokens)
                self.__documents.append(doc)
        if not all_tokens or not isinstance(all_tokens, list):
            raise ValueError
        self._vectorizer.set_tokenized_corpus(all_tokens)
        self._vectorizer.build()
        for index, tokens in enumerate(all_tokens):
            self.__vectors[index] = self._vectorizer.vectorize(tokens)

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
        result = []
        for index in indices:
            if index not in result:
                result.append(index)
        return [(index, self.__vectors[index]) for index in result]

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
        result = []
        for index in indices:
            if self.__documents[index] not in result:
                result.append(self.__documents[index])
        return result


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
        if not isinstance(query, str) or not query\
                or not isinstance(n_neighbours, int) or n_neighbours <= 0:
            raise ValueError
        tokenized_query = self._tokenizer.tokenize(query)
        if not isinstance(tokenized_query, list):
            raise ValueError
        vectorized_query = self._db.get_vectorizer().vectorize(tokenized_query)
        if not vectorized_query:
            raise ValueError
        docs_vectors = [vector[1] for vector in self._db.get_vectors()]
        if len(docs_vectors[0]) < len(vectorized_query):
            raise ValueError
        distances = self._calculate_knn(vectorized_query, docs_vectors, n_neighbours)
        if not distances:
            raise ValueError
        documents = self._db.get_raw_documents(tuple(index for index in range(n_neighbours)))
        return [(dist, documents[ind]) for ind, dist in distances]


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
        if not isinstance(index, int) or index < 0 or index is None:
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
        self.__clusters = []
        self._db = db
        self._n_clusters = n_clusters

    def train(self) -> None:
        """
        Train k-means algorithm.
        """
        centroids = self._db.get_vectors()[:self._n_clusters]
        for centroid in centroids:
            self.__clusters.append(ClusterDTO(centroid[-1]))
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
        doc_vectors = self._db.get_vectors()
        for index, vector in doc_vectors:
            distances = []
            for centroid in centroids:
                distance = calculate_distance(vector, centroid)
                if distance is None:
                    raise ValueError
                distances.append(distance)
            nearest = min(distances)
            self.__clusters[distances.index(nearest)].add_document_index(index)
        for cluster in self.__clusters:
            current_vectors = [doc_vectors[index][-1] for index in cluster.get_indices()]
            new_centroid = [sum(vector[ind] for ind in range(len(vector))) / len(current_vectors)
                            for vector in current_vectors]
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
        if (not isinstance(query_vector, tuple) or not query_vector or not n_neighbours
                or not isinstance(n_neighbours, int)):
            raise ValueError
        distances_for_clusters = []
        for cluster in self.__clusters:
            distance = calculate_distance(query_vector, cluster.get_centroid())
            if distance is None:
                raise ValueError
            distances_for_clusters.append(distance)
        nearest_index = distances_for_clusters.index(min(distances_for_clusters))
        current_cluster = self.__clusters[nearest_index]
        if not current_cluster.get_centroid():
            current_cluster = self.__clusters[0]
        indices = current_cluster.get_indices()
        vectors = self._db.get_vectors(indices)
        distances_for_vectors = []
        for index, vector in vectors:
            distance = calculate_distance(query_vector, vector)
            if not isinstance(distance, float) or distance is None:
                raise ValueError
            distances_for_vectors.append((distance, index))
        return sorted(distances_for_vectors, key=lambda x: x[0])[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int| list[str]]]: List with information about each cluster
        """
        if not isinstance(num_examples, int) or not num_examples or num_examples < 0:
            raise ValueError
        result: list[dict[str, int | list[str]]] = []
        for cluster in self.__clusters:
            cluster_vectors = self._db.get_vectors(cluster.get_indices())
            distances = [(calculate_distance(vector[1], cluster.get_centroid()), index)
                         for vector in cluster_vectors for index in cluster.get_indices()]
            if distances is None or any(distance[0] is None for distance in distances):
                raise ValueError
            closest_docs_ind = [pair[1] for pair in distances[:num_examples]]
            closest_docs = self._db.get_raw_documents(tuple(closest_docs_ind))
            result.append({
                'cluster_id': self.__clusters.index(cluster),
                'documents': closest_docs
            })
        return result

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        sum_for_cluster = []
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            documents_ind = cluster.get_indices()
            documents = [pair[1] for pair in self._db.get_vectors(documents_ind)]
            sum_for_pairs = []
            for doc in documents:
                sum_for_pairs.append(sum((centroid[ind] - doc[ind]) ** 2
                                         for ind in range(len(centroid))))
            sum_for_cluster.append(sum(num for num in sum_for_pairs))
        return sum(number for number in sum_for_cluster)

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
        if not isinstance(new_clusters, list) or new_clusters is None \
                or not isinstance(threshold, float):
            raise ValueError
        if not self.__clusters:
            raise ValueError
        for index, cluster in enumerate(new_clusters):
            old_centroid = self.__clusters[index].get_centroid()
            new_centroid = cluster.get_centroid()
            distance = calculate_distance(old_centroid, new_centroid)
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
        if (not isinstance(query, str) or not query or not isinstance(n_neighbours, int)
                or not n_neighbours):
            raise ValueError
        query_tokens = self._db.get_tokenizer().tokenize(query)
        if query_tokens is None:
            raise ValueError
        query_vector = self._db.get_vectorizer().vectorize(query_tokens)
        if query_vector is None:
            raise ValueError
        self.__algo.train()
        pairs = self.__algo.infer(query_vector, n_neighbours)
        if pairs is None:
            raise ValueError
        distances = [pair[0] for pair in pairs]
        indices = [pair[1] for pair in pairs]
        documents = self._db.get_raw_documents(tuple(indices))
        result = []
        for index in indices:
            result.append((distances[index], documents[index]))
        return result

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        if not isinstance(num_examples, int) or not num_examples or num_examples < 0\
                or not isinstance(output_path, str) or not output_path:
            raise ValueError
        data = self.__algo.get_clusters_info(num_examples)
        if data is None:
            raise ValueError
        with open(output_path, "w", encoding="utf-8") as document:
            json.dump(data, document)

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
