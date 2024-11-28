"""
Lab 4.

Vector search with clusterization
"""
# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called

import json
from copy import copy

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
    if not (isinstance(text, str) and len(text) > 0):
        raise ValueError
    return text.split("\n")


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
        Vectorizer.__init__(self, self._corpus)
        self._avg_doc_len = -1.0

    def set_tokenized_corpus(self, tokenized_corpus: TokenizedCorpus) -> None:
        """
        Set tokenized corpus and average document length.

        Args:
            tokenized_corpus (TokenizedCorpus): Tokenized texts corpus.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not (isinstance(tokenized_corpus, list) and len(tokenized_corpus) > 0):
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
        if not (isinstance(tokenized_document, list) and len(tokenized_document) > 0):
            raise ValueError
        bm25_vector = self._calculate_bm25(tokenized_document)
        if bm25_vector is None:
            raise ValueError
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
        if not (isinstance(self._vocabulary, list) and isinstance(self._idf_values, dict)):
            raise ValueError
        if not (len(self._vocabulary) > 0 and len(self._idf_values) > 0):
            return ()
        vector_to_fill = [0.0] * len(self._vocabulary)
        bm25 = calculate_bm25(self._vocabulary, tokenized_document,
                              self._idf_values, avg_doc_len=self._avg_doc_len,
                              doc_len=len(tokenized_document))
        if bm25 is None:
            raise ValueError
        for word in bm25:
            vec_ind = self._token2ind.get(word, -1)
            if not isinstance(vec_ind, int):
                raise ValueError
            if not vec_ind == -1:
                vector_to_fill[vec_ind] = bm25[word]
        return tuple(vector_to_fill)


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
        if not isinstance(stop_words, list):
            raise ValueError
        self.__vectors = {}
        self.__documents = []
        self._tokenizer = Tokenizer(stop_words)
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
        if not isinstance(corpus, list):
            raise ValueError
        self.__documents = corpus
        tokenized_documents = []
        for document in corpus:
            document_tokens_list = self._tokenizer.tokenize(document)
            if not isinstance(document_tokens_list, list):
                raise ValueError
            if not len(document_tokens_list) == 0:
                tokenized_documents.append(document_tokens_list)
        self._vectorizer.set_tokenized_corpus(tokenized_documents)
        self._vectorizer.build()
        for index, token_document in enumerate(tokenized_documents):
            document_vector = self._vectorizer.vectorize(token_document)
            if not isinstance(document_vector, tuple):
                raise ValueError
            self.__vectors[index] = document_vector

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
        unique_indices = []
        for index in indices:
            if not index in unique_indices:
                unique_indices.append(index)
        return [self.__documents[index] for index in unique_indices]


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
        BasicSearchEngine.__init__(self, self._db.get_vectorizer(), self._db.get_tokenizer())

    def retrieve_relevant_documents(self, query: str, n_neighbours: int) -> list[tuple[float, str]]:
        """
        Get relevant documents.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]]: Relevant documents with their distances.
        """
        if not (isinstance(query, str) and isinstance(n_neighbours, int) and
                len(query) > 0 and n_neighbours > 0):
            raise ValueError
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if not isinstance(tokenized_query, list):
            raise ValueError
        query_vector = self._db.get_vectorizer().vectorize(tokenized_query)
        vectors_with_indices = self._db.get_vectors()
        vectors_wo_indexes = [pair[1] for pair in vectors_with_indices]
        nearest_docs = self._calculate_knn(query_vector, vectors_wo_indexes, n_neighbours)
        if not (isinstance(nearest_docs, list) and len(nearest_docs) > 0):
            raise ValueError
        true_indices = tuple(vectors_with_indices[pair[0]][0] for pair in nearest_docs)
        retrieved_documents = self._db.get_raw_documents(true_indices)
        return_list = []
        for index, document in enumerate(nearest_docs):
            return_list.append((document[1], retrieved_documents[index]))
        return return_list


class ClusterDTO:
    """
    Store clusters.
    """

    _centroid: Vector
    __indices: list[int]

    def __init__(self, centroid_vector: Vector) -> None:
        """
        Initialize an instance of the ClusterDTO class.

        Args:
            centroid_vector (Vector): Centroid vector.
        """
        self._centroid = centroid_vector
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
        return self._centroid

    def set_new_centroid(self, new_centroid: Vector) -> None:
        """
        Set new centroid for cluster.

        Args:
            new_centroid (Vector): New centroid vector.

        Raises:
            ValueError: In case of inappropriate type input arguments,
                or if input arguments are empty.
        """
        if not (isinstance(new_centroid, tuple) and len(new_centroid) > 0):
            raise ValueError
        self._centroid = new_centroid

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
        if not (isinstance(index, int) and index >= 0):
            raise ValueError
        if not index in self.__indices:
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
        self.__clusters = [ClusterDTO(pair[1]) for pair in centroids]
        while True:
            new_clusters = self.run_single_train_iteration()
            if self._is_convergence_reached(new_clusters):
                break
        self.run_single_train_iteration()

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        clusters = copy(self.__clusters)
        for cluster in clusters:
            cluster.erase_indices()
        db_vectors = self._db.get_vectors()
        for index, vector in db_vectors:
            distance_list = []
            for cluster_index, cluster in enumerate(clusters):
                centroid_distance = calculate_distance(cluster.get_centroid(), vector)
                if not isinstance(centroid_distance, float):
                    raise ValueError
                distance_list.append((cluster_index, centroid_distance))
            min_distance_index = min(distance_list, key=lambda a: a[1])[0]
            clusters[min_distance_index].add_document_index(index)
        for cluster in clusters:
            vector_sums = [0.0] * len(cluster.get_centroid())
            for vector_index in cluster.get_indices():
                vector_from_index = db_vectors[vector_index][1]
                if not len(vector_from_index) == len(vector_sums):
                    raise ValueError
                for mean_vector_index, _ in enumerate(vector_sums):
                    vector_sums[mean_vector_index] += vector_from_index[mean_vector_index]
            mean_vector = tuple(value / len(cluster) for value in vector_sums)
            cluster.set_new_centroid(mean_vector)
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
        if not (isinstance(query_vector, tuple) and isinstance(n_neighbours, int)):
            raise ValueError
        centroid_distances = []
        for index, cluster in enumerate(self.__clusters):
            centroid_distance = calculate_distance(query_vector, cluster.get_centroid())
            if not isinstance(centroid_distance, float):
                raise ValueError
            centroid_distances.append((index, centroid_distance))
        min_distance_index = min(centroid_distances, key=lambda a: a[1])[0]
        closest_cluster = self.__clusters[min_distance_index]
        index_vectors = self._db.get_vectors(closest_cluster.get_indices())
        vector_distances = []
        for index, vector in index_vectors:
            vector_distance = calculate_distance(query_vector, vector)
            if not isinstance(vector_distance, float):
                raise ValueError
            vector_distances.append((vector_distance, index))
        return sorted(vector_distances, key=lambda a: a[0])[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int | list[str]]]: List with information about each cluster
        """
        if not isinstance(num_examples, int):
            raise ValueError
        list_of_cluster_info = []
        for cluster_index, cluster in enumerate(self.__clusters):
            info_dict: dict[str, int | list[str]] = {"id": cluster_index}
            cluster_vectors = self._db.get_vectors(cluster.get_indices())
            vector_distances = []
            for index, vector in cluster_vectors:
                vector_distance = calculate_distance(cluster.get_centroid(), vector)
                if not isinstance(vector_distance, float):
                    raise ValueError
                vector_distances.append((index, vector_distance))
            vector_distances = sorted(vector_distances, key=lambda a: a[1])[:num_examples]
            doc_indices = tuple(pair[0] for pair in vector_distances)
            info_dict["nearest_docs"] = self._db.get_raw_documents(doc_indices)
            list_of_cluster_info.append(info_dict)
        return list_of_cluster_info

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        sse = 0.0
        db_vectors = self._db.get_vectors()
        for cluster in self.__clusters:
            centroid = cluster.get_centroid()
            for vector_index in cluster.get_indices():
                vector_from_index = db_vectors[vector_index][1]
                sse += sum((centroid[index] - element) ** 2
                           for index, element in enumerate(vector_from_index))
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
        if not (isinstance(new_clusters, list) and isinstance(threshold, float) and
                len(new_clusters) > 0):
            raise ValueError
        for index, old_cluster in enumerate(self.__clusters):
            centroid_distance = calculate_distance(old_cluster.get_centroid(),
                                                   new_clusters[index].get_centroid())
            if not isinstance(centroid_distance, float):
                raise ValueError
            if centroid_distance >= threshold:
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
        if not (isinstance(query, str) and isinstance(n_neighbours, int) and
                len(query) > 0 and n_neighbours > 0):
            raise ValueError
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if not isinstance(tokenized_query, list):
            raise ValueError
        query_vector = self._db.get_vectorizer().vectorize(tokenized_query)
        if not isinstance(query_vector, tuple):
            raise ValueError
        nearest_docs = self.__algo.infer(query_vector, n_neighbours)
        if not (isinstance(nearest_docs, list) and len(nearest_docs) > 0):
            raise ValueError
        doc_indices = tuple(pair[1] for pair in nearest_docs)
        retrieved_documents = self._db.get_raw_documents(doc_indices)
        return_list = []
        for index, document in enumerate(nearest_docs):
            return_list.append((document[0], retrieved_documents[index]))
        return return_list

    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        with open(output_path, 'w', encoding='utf-8') as file_to_save:
            json.dump(self.__algo.get_clusters_info(num_examples), file_to_save)

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
        self._engine.index_documents(db.get_raw_documents())

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
        engine = SearchEngine(db.get_vectorizer(), db.get_tokenizer())
        VectorDBEngine.__init__(self, db, engine)
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
        engine = AdvancedSearchEngine(db.get_vectorizer(), db.get_tokenizer())
        VectorDBEngine.__init__(self, db, engine)
        self._engine.index_documents(db.get_raw_documents())
