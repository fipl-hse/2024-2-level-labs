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
        raise ValueError("Empty string")
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
        if not tokenized_corpus:
            raise ValueError("Empty tokenized_corpus")
        if self._corpus:
            self._corpus = []
        summary_len = 0
        for token in tokenized_corpus:
            summary_len += len(token)
            self._corpus.append(token)
        self._avg_doc_len = summary_len / len(self._corpus)

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
            raise ValueError("Empty tokenized_document")
        vector = self._calculate_bm25(tokenized_document)
        if vector is None:
            raise ValueError('Vector is None')
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
        vector_bm_25 = [0.0] * len(self._vocabulary)

        bm_25 = calculate_bm25(self._vocabulary, tokenized_document,
                        self._idf_values, 1.5, 0.75,
                        self._avg_doc_len, len(tokenized_document))
        for i, token in enumerate(self._vocabulary):
            if token in tokenized_document:
                vector_bm_25[i] = bm_25.get(token, 0.0)
        return tuple(vector_bm_25)


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
            raise ValueError('Empty corpus')
        self._vectorizer.build()
        self.__documents = corpus
        tokenized_corpus = []
        for text in corpus:
            tokens = self._tokenizer.tokenize(text)
            if not isinstance(tokens, list):
                raise ValueError('Tokens is not list')
            tokenized_corpus.append(tokens)
        if None in tokenized_corpus:
            raise ValueError('NoneType in tokenized_corpus')
        self._vectorizer.set_tokenized_corpus(tokenized_corpus)
        for i, tokenized_text in enumerate(tokenized_corpus):
            if not tokenized_text:
                tokenized_corpus.pop(i)
                self.__documents.pop(i)
                continue
            self.__vectors[i] = self._vectorizer.vectorize(tokenized_text)
        if None in self.__vectors.values():
            raise ValueError('NoneType in vectors')

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
        vectors = []
        if indices is None:
            for key, value in self.__vectors.items():
                vectors.append((key, value))
            return vectors
        for key, value in self.__vectors.items():
            if key not in indices:
                continue
            vectors.append((key, value))
        return vectors

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
        docs = []
        for i, text in enumerate(self.__documents):
            if i not in indices:
                continue
            docs.append(text)
        return docs


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
        if not query or not isinstance(query, str)\
                or not isinstance(n_neighbours, int) or not n_neighbours > 0:
            raise ValueError
        self._db.get_vectorizer().build()
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if not tokenized_query:
            raise ValueError('Empty tokenized_query')
        query_vector = self._db.get_vectorizer().vectorize(tokenized_query)
        if not query_vector:
            raise ValueError('Empty query_vector')
        self._db.put_corpus(self._db.get_raw_documents())
        docs_vectors = [vec[1] for vec in self._db.get_vectors()]
        relevant_documents = self._calculate_knn(query_vector, docs_vectors, n_neighbours)
        if relevant_documents is None:
            raise ValueError
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
        if not new_centroid:
            raise ValueError('Empty new_centroid')
        self.__centroid = new_centroid

    def erase_indices(self) -> None:
        """
        Clear indexes.
        """
        if len(self.__indices) > 0:
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
            raise ValueError('Value error index')
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
        for centroid in new_centroids:
            self.__clusters.append(ClusterDTO(centroid[1]))
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
            centroids.append(cluster.get_centroid())
        if None in centroids:
            raise ValueError('NoneType in centroids')
        vectors = self._db.get_vectors()
        for i, vector in self._db.get_vectors():
            distances = []
            for centroid in centroids:
                distance = calculate_distance(centroid, vector)
                if distance is None:
                    raise ValueError('Distance is NoneType')
                distances.append(distance)
            self.__clusters[distances.index(min(distances))].add_document_index(i)
        for cluster in self.__clusters:
            cl_vectors = [vectors[index][1] for index in cluster.get_indices()]
            new_centroid = []
            for vector in zip(*cl_vectors):
                value = sum(vector) / len(cl_vectors)
                new_centroid.append(value)
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
        if not query_vector or not n_neighbours or\
                not isinstance(query_vector, tuple) or not isinstance(n_neighbours, int):
            raise ValueError('Wrong type arguments or empty arguments')
        distances = []
        for cluster in self.__clusters:
            distance = calculate_distance(query_vector, cluster.get_centroid())
            if distance is None:
                raise ValueError('Distance is NoneType')
            distances.append(distance)
        near_cluster = self.__clusters[distances.index(min(distances))]
        if not near_cluster.get_centroid():
            near_cluster = self.__clusters[0]
        indices = near_cluster.get_indices()
        if indices is None:
            raise ValueError('Indices are NoneType')
        cl_vectors = self._db.get_vectors(indices)
        if cl_vectors is None:
            raise ValueError('Cluster vectors are NoneType')
        n_distances = []
        for i, vector in cl_vectors:
            distance = calculate_distance(query_vector, vector)
            if distance is None:
                raise ValueError('Distance is NoneType')
            n_distances.append((distance, i))
        n_distances.sort(key=lambda x: x[0])
        return n_distances[:n_neighbours]

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
            raise ValueError('Empty new_clusters')
        for i, cluster in enumerate(self.__clusters):
            distance = calculate_distance(cluster.get_centroid(),
                                          new_clusters[i].get_centroid())
            if distance is None:
                raise ValueError('Distance is NoneType')
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
        if not query or not isinstance(query, str) or\
                not isinstance(n_neighbours, int) or not n_neighbours > 0:
            raise ValueError('Bad input')
        token_query = self._db.get_tokenizer().tokenize(query)
        if token_query is None:
            raise ValueError('Query token is NoneType')
        vector_query = self._db.get_vectorizer().vectorize(token_query)
        if vector_query is None:
            raise ValueError('Query vector is NoneType')
        self.__algo.train()
        infer = self.__algo.infer(vector_query, n_neighbours)
        if infer is None:
            raise ValueError('Infer is NoneType')
        indices = [dist_ind[1] for dist_ind in infer]
        relevant_docs = self._db.get_raw_documents(tuple(indices))
        return [(infer[i][0], relevant_docs[i]) for i in range(len(infer))]

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
