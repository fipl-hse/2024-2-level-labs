"""
Lab 4.

Vector search with clusterization
"""

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called
from lab_2_retrieval_w_bm25.main import calculate_bm25
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
        if (tokenized_corpus is None
                or not tokenized_corpus
                or not isinstance(tokenized_corpus, list)
                or not all(isinstance(tok_paragraph, list) for tok_paragraph in tokenized_corpus)):
            raise ValueError('Invalid input')
        self._corpus = tokenized_corpus

        self._avg_doc_len = sum(len(every_paragraph)
                                for every_paragraph in self._corpus) / len(self._corpus)

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
        if (not tokenized_document
                or not isinstance(tokenized_document, list)
                or not all(isinstance(el, str) for el in tokenized_document)):
            raise ValueError('Invalid input')

        bm25_vector = self._calculate_bm25(tokenized_document)
        if not bm25_vector:
            raise ValueError('The function returned an empty Vector')

        return bm25_vector

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
        if (not tokenized_document
                or not isinstance(tokenized_document, list)
                or not all(isinstance(elem, str) for elem in tokenized_document)):
            raise ValueError('Invalid input')

        bm25_vector = [0.0] * len(self._vocabulary)

        bm25 = calculate_bm25(self._vocabulary,
                              tokenized_document,
                              self._idf_values,
                              avg_doc_len=self._avg_doc_len, doc_len=len(tokenized_document))
        for i, token in enumerate(self._vocabulary):
            if bm25 is not None:
                bm25_vector[i] = bm25[token]
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
        if (not corpus or not isinstance(corpus, list)
                or not all(isinstance(element, str) for element in corpus)):
            raise ValueError('Invalid input')

        list_of_tok_paragraphs = []
        for doc_aka_str in corpus:
            doc_aka_list = self._tokenizer.tokenize(doc_aka_str)
            if doc_aka_list:
                list_of_tok_paragraphs.append(doc_aka_list)
                self.__documents.append(doc_aka_str)
        if not list_of_tok_paragraphs:
            raise ValueError('The function returned an empty list')

        self._vectorizer.set_tokenized_corpus(list_of_tok_paragraphs)
        self._vectorizer.build()
        for index_of_paragraph, tok_paragraph in enumerate(list_of_tok_paragraphs):
            self.__vectors[index_of_paragraph] = self._vectorizer.vectorize(tok_paragraph)

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
        final_ind = []
        for index in indices:
            if index not in final_ind:
                final_ind.append(index)
        return [(index, self.__vectors[index]) for index in final_ind]

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
        corpus_aka_list = []
        for doc_ind in indices:
            if self.__documents[doc_ind] not in corpus_aka_list:
                corpus_aka_list.append(self.__documents[doc_ind])
        return corpus_aka_list


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
                or not isinstance(n_neighbours, int)
                or n_neighbours < 0):
            raise ValueError('Invalid input')

        tok_query = self._tokenizer.tokenize(query)
        if tok_query is None:
            raise ValueError('tokenize() returned None')

        vec_query = self._vectorizer.vectorize(tok_query)
        if vec_query is None:
            raise ValueError('vectorize() returned None')

        c_knn_result = self._calculate_knn(vec_query,
                                           [vec[1] for vec in self._db.get_vectors()], n_neighbours)
        if not c_knn_result:
            raise ValueError('The function returned an empty list of tuples')
        get_raw_docs_result = self._db.get_raw_documents(tuple(t[0] for t in c_knn_result))
        if not get_raw_docs_result:
            raise ValueError('The function returned an empty list')

        return [(c_knn_result[ind][1], s) for ind, s in enumerate(get_raw_docs_result)]


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
        if (new_centroid is None or not new_centroid
                or not isinstance(new_centroid, tuple)):
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
        if not isinstance(index, int) or index < 0 or index is None:
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
        self.__clusters = []
        self._db = db
        self._n_clusters = n_clusters

    def train(self) -> None:
        """
        Train k-means algorithm.
        """
        first_n_centroids = self._db.get_vectors()[:self._n_clusters]
        for centroid in first_n_centroids:
            self.__clusters.append(ClusterDTO(centroid[1]))
        self.run_single_train_iteration()
        while not self._is_convergence_reached(self.__clusters):
            self.run_single_train_iteration()

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        new_centroids = []
        for every_cluster in self.__clusters:
            every_cluster.erase_indices()
            new_centroids.append(every_cluster.get_centroid())

        for index, vector_tuple in self._db.get_vectors():
            distances = []
            for centroid in new_centroids:
                dist_between_vec_from_db_to_centroid_of_cluster = calculate_distance(vector_tuple,
                                                                                     centroid)
                if dist_between_vec_from_db_to_centroid_of_cluster is None:
                    raise ValueError('calculate_distance() returned None')
                distances.append(dist_between_vec_from_db_to_centroid_of_cluster)
            self.__clusters[distances.index(min(distances))].add_document_index(index)

        for cluster in self.__clusters:
            current_vectors = [self._db.get_vectors()[index][-1] for index in cluster.get_indices()]
            cluster.set_new_centroid(tuple(sum(row) / len(current_vectors) for row
                                           in zip(*current_vectors)))

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
        if (not query_vector or query_vector is None
                or n_neighbours < 0 or not isinstance(n_neighbours, int)):
            raise ValueError('Invalid input')

        dist_between_q_vec_and_cluster = []
        for every_cluster in self.__clusters:
            if every_cluster.get_centroid() is None:
                continue
            dist_between = calculate_distance(query_vector, every_cluster.get_centroid())
            if dist_between is None:
                raise ValueError('calculate_distance() returned None')
            dist_between_q_vec_and_cluster.append(dist_between)

        chosen_cluster = self.__clusters[
            dist_between_q_vec_and_cluster.index(min(dist_between_q_vec_and_cluster))]
        if chosen_cluster is None:
            raise ValueError('chosen_cluster is None')

        if chosen_cluster.get_indices() is None:
            raise ValueError('get_indices() returned None')
        final_list = []
        for vec_index, vec in self._db.get_vectors(chosen_cluster.get_indices()):
            distance_again = calculate_distance(query_vector, vec)
            if distance_again is None:
                raise ValueError('calculate_distance() returned None')
            final_list.append((distance_again, vec_index))

        return sorted(final_list, key=lambda x: x[0])[:n_neighbours]

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
        if (not new_clusters or not isinstance(new_clusters, list)
                or not isinstance(threshold, float) or threshold <= 0):
            raise ValueError('Invalid input')

        for cl_index, cluster in enumerate(new_clusters):
            current_centroid = cluster.get_centroid()
            previous_centroid = self.__clusters[cl_index].get_centroid()
            difference = calculate_distance(previous_centroid, current_centroid)
            if difference is None:
                raise ValueError('calculate_distance() returned None')
            if difference > threshold:
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
        if (not isinstance(query, str) or query is None or not query
                or n_neighbours <= 0 or not isinstance(n_neighbours, int)):
            raise ValueError('Invalid input')

        tok_query = self._db.get_tokenizer().tokenize(query)
        if tok_query is None:
            raise ValueError('_db.get_tokenizer().tokenize() returned None')
        vec_query = self._db.get_vectorizer().vectorize(tok_query)
        if vec_query is None:
            raise ValueError('_db.get_vectorizer().vectorize() returned None')

        self.__algo.train()

        t_dist_and_vec_index = self.__algo.infer(vec_query, n_neighbours)
        if t_dist_and_vec_index is None:
            raise ValueError('infer() returned None')

        l_with_str_aka_docs = self._db.get_raw_documents(tuple(
            every_tuple[1] for every_tuple in t_dist_and_vec_index))
        if l_with_str_aka_docs is None:
            raise ValueError('get_raw_documents() returned None')

        rel_docs_with_indices = []
        t_dist_and_vec_index = [t[0] for t in self.__algo.infer(vec_query, n_neighbours)]
        indices = [pair[1] for pair in self.__algo.infer(vec_query, n_neighbours)]
        for index, doc in enumerate(self._db.get_raw_documents(tuple(indices))):
            rel_docs_with_indices.append((t_dist_and_vec_index[index], doc))

        return rel_docs_with_indices

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
