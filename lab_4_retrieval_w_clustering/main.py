"""
Lab 4.

Vector search with clusterization
"""

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called
from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vector, Vectorizer, calculate_distance
from lab_2_retrieval_w_bm25.main import calculate_idf, calculate_bm25
import math
import re

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
        raise ValueError
    text = text.replace('\n', ' ')
    paragraphs = re.split(r'\s{2,}', text.strip())
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


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
        super().__init__(self._corpus)
        self._corpus = []
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
            raise ValueError
        self._corpus = tokenized_corpus
        self._avg_doc_len = sum(len(paragraph) for paragraph in tokenized_corpus) / len(tokenized_corpus)

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
            raise ValueError
        if not self._vocabulary:
            return ()
        if not self._calculate_bm25(tokenized_document):
            raise ValueError
        return self._calculate_bm25(tokenized_document)

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
            raise ValueError
        vector = [0.0 for _ in range(len(self._vocabulary))]
        document_len = len(tokenized_document)
        idf_value = self._idf_values
        dict_bm25 = calculate_bm25(self._vocabulary, tokenized_document,
                                   idf_value, avg_doc_len=self._avg_doc_len, doc_len=document_len)
        if dict_bm25 is None:
            raise ValueError('_calculate_bm25 returned None, please check implemention')
        for i, word in self._token2ind:
            vector[i] = dict_bm25.get(word, 0.0)
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
        self.stop_words = stop_words
        self._vectorizer = BM25Vectorizer()
        self._tokenizer = Tokenizer(self.stop_words)
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
        if not isinstance(corpus, list) or not corpus:
            raise ValueError("Input arguments are wrong")
        tokenized_documents = self._tokenizer.tokenize_documents(corpus)
        if tokenized_documents is None:
            raise ValueError("Tokenization returned None")

        filtered_documents = [self._tokenizer._remove_stop_words(paragraph) for paragraph in
                              tokenized_documents if self._tokenizer._remove_stop_words(paragraph)]
        if not filtered_documents:
            raise ValueError("No valid documents after removing stopwords")

        self._vectorizer.set_tokenized_corpus(tokenized_documents)
        self.__vectors = {index: self._vectorizer._calculate_bm25(doc) for index, doc
                          in enumerate(filtered_documents)}
        if not self.__vectors:
            raise ValueError("Vectorization returned None")
        self.__documents = [corpus[i] for i in range(len(corpus)) if tokenized_documents[i]]

    def get_vectorizer(self) -> BM25Vectorizer:
        """
        Get an object of the BM25Vectorizer class.

        Returns:
            BM25Vectorizer: BM25Vectorizer class object.
        """
        new_object = BM25Vectorizer()
        return new_object

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
        else:
            return [(index, self.__vectors[index]) for index in indices if index in self.__vectors]


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
        else:
            indices_set = set(indices)
            real_docs = [doc for ind, doc in enumerate(self.__documents) if ind in indices_set]
            return real_docs


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
        super().__init__(self._vectorizer, self._tokenizer)
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
        if not query or n_neighbours <=0:
            raise ValueError("Query must be a non-empty string and n_neighbours "
                             "must be a positive integer")
        query_list = self._tokenizer.tokenize(query)
        query_vector = self._db.get_vectorizer()._calculate_bm25(query_list)
        if query_vector is None:
            raise ValueError("Query vector calculation returned None")
        document_vectors = self._db.get_vectors()
        if document_vectors is None:
            raise ValueError("Document vectors retrieval returned None")
        vectors = [vector for _, vector in document_vectors]
        knn_indices = self._calculate_knn(query_vector, vectors, n_neighbours)
        if knn_indices is None:
            raise ValueError("KNN returned None")

        indices = tuple([ind for ind, distance in knn_indices])

        relevant_docs = [(distance, self._db.get_raw_documents(indices)[0]) for distance, index in knn_indices]
        return relevant_docs


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
            raise ValueError("Incorrect new centroid")
        self.__centroid = new_centroid

    def erase_indices(self) -> None:
        """
        Clear indexes.
        """
        self.__indices.clear()

    def add_document_index(self, index: int) -> None:
        """
        Add document index.

        Args:
            index (int): Index of document.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty.
        """
        if not index or index < 0:
            raise ValueError("Index must not be empty or negative")
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
        starting_centroids = self.__clusters[self._n_clusters]
        document_vectors = self._db.get_vectors()
        if len(document_vectors) < self._n_clusters:
            raise ValueError("Lack of clusters")

        for vector in document_vectors:
            if not isinstance(vector, tuple) or not all(isinstance(x, float) for x in vector):
                raise ValueError(f"Invalid vector structure: {vector}")

        self.__clusters = [ClusterDTO(centroid_vector=document_vectors[i][1]) for i in range(self._n_clusters)]

        check = False

        while not check:
            new_clusters = self.run_single_train_iteration()
            check = True
            for old_cluster, new_cluster in zip(self.__clusters, new_clusters):
                if old_cluster.centroid_vector != new_cluster.centroid_vector:
                    check = False
                    break

            if not check:
                self.__clusters = new_clusters

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        for cluster in self.__clusters:
            cluster.erase_indices()
        document_vectors = self._db.get_vectors()
        if not document_vectors:
            raise ValueError("We can't get vectors")

        cluster_document = {cluster: [] for cluster in self.__clusters}

        for index, doc in enumerate(document_vectors):
            closest_cluster = None
            closest_distance = float('information for loop')

            for cluster in self.__clusters:
                centroid = cluster.get_centroid()
                if not centroid:
                    raise ValueError("Centroid is empty")
                distance = math.sqrt(sum((value - doc[i]) ** 2 for i, value in enumerate(centroid)))

                if distance < closest_distance:
                    closest_distance = distance
                    closest_cluster = cluster

                if closest_cluster is not None:
                    closest_cluster.add_document_index(index)
                    cluster_document[closest_cluster].append(doc)

        for cluster, docs in cluster_document.items():
            if docs:
                average_value = tuple(sum(values) / len(docs) for values in zip(*docs))
                if not cluster.set_new_centroid(average_value):
                    raise ValueError("We can't set new centroid")

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
        if not query_vector or query_vector is None:
            raise ValueError("Check, please, query vector input in infer")
        if not n_neighbours or n_neighbours < 0:
            raise ValueError("Check, please, n_neighbours input in infer")
        dict_for_clusters = {}
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            distance = calculate_distance(query_vector, centroid)
            dict_for_clusters[cluster] = distance
        min_cluster = min(dict_for_clusters, key=dict_for_clusters.get)
        indices = min_cluster.get_indices()
        vectors = self._db.get_vectors(indices)
        list_for_vectors = []
        for vector in vectors:
            distance = calculate_distance(query_vector, vector[1])
            list_for_vectors.append((distance, vector[0]))
        sorted_list = sorted(list_for_vectors, reverse=True, key=lambda x: x[0])
        return sorted_list[:n_neighbours]

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
        if not new_clusters or not threshold:
            raise ValueError("Incorrect input data in _is_convergence_reached")
        for cluster in self.__clusters:
            for clust in new_clusters:
                old_centroid = cluster.get_centroid()
                new_centroid = clust.get_centroid()
                if old_centroid is None or new_centroid is None:
                    raise ValueError("Please, check centroids in clusters")
                if calculate_distance(old_centroid, new_centroid) < threshold:
                    return True
                else:
                    return False


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
        if not query or not n_neighbours:
            raise ValueError("Please, check the input in retrieving relevant documents")


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
