"""
Lab 4.

Vector search with clusterization
"""
import json

from lab_2_retrieval_w_bm25.main import calculate_bm25

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called
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
    if not text or not isinstance(text,str):
        raise ValueError('Get_paragraphs input error')
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
        if not tokenized_corpus or not isinstance(tokenized_corpus,list):
            raise ValueError('bm25_set_tokenized_corpus input error')
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
        if not tokenized_document or not isinstance(tokenized_document,list)\
                or not all(isinstance(paragraph,str) for paragraph in tokenized_document):
            raise ValueError('bm25_vectorize input error')
        vectorized = self._calculate_bm25(tokenized_document)
        if not vectorized:
            raise ValueError('bm25_vectorize vectorization error')
        return vectorized

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
        if not tokenized_document or not isinstance(tokenized_document,list) or not\
                all(isinstance(paragraph,str) for paragraph in tokenized_document):
            raise ValueError('bm25_calculate_bm25 input error')
        vector_list = [0.0] * len(self._vocabulary)
        bm25 = calculate_bm25(vocab=self._vocabulary, document=tokenized_document,
                              idf_document=self._idf_values,
                              avg_doc_len=self._avg_doc_len, doc_len=len(tokenized_document))
        if not bm25:
            return tuple(vector_list)
        for index, word in enumerate(self._vocabulary):
            vector_list[index] = bm25[word]
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
            raise ValueError('DB_put_corpus input error')
        tokenized_docs = []
        for document in corpus:
            tokenized_doc = self._tokenizer.tokenize(document)
            if tokenized_doc:
                self.__documents.append(document)
                tokenized_docs.append(tokenized_doc)
        if not tokenized_docs:
            raise ValueError('DB_put_corpus tokenized_docs error')
        self._vectorizer.set_tokenized_corpus(tokenized_docs)
        self._vectorizer.build()
        for doc in tokenized_docs:
            doc_index = tokenized_docs.index(doc)
            vectorized = self._vectorizer.vectorize(doc)
            self.__vectors[doc_index] = vectorized

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
        needed_vectors = []
        for i in indices:
            needed_vectors.append((i, self.__vectors[i]))
        return needed_vectors


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
        repeats = []
        needed_docs = []
        for i in indices:
            if not i in repeats:
                needed_docs.append(self.__documents[i])
                repeats.append(i)
        return needed_docs


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
        super().__init__(self._db.get_vectorizer(),self._db.get_tokenizer())

    def retrieve_relevant_documents(self, query: str, n_neighbours: int) -> list[tuple[float, str]]:
        """
        Get relevant documents.

        Args:
            query (str): Query for obtaining relevant documents.
            n_neighbours (int): Number of relevant documents to return.

        Returns:
            list[tuple[float, str]]: Relevant documents with their distances.
        """
        if not query or not isinstance(query, str) or not n_neighbours\
                or not isinstance(n_neighbours,int) or n_neighbours <= 0:
            raise ValueError('VectorDBSearchEngine retrieve_relevant_documents input error')
        tokenized_query = self._tokenizer.tokenize(query)
        if not tokenized_query:
            raise ValueError('VectorDBSearchEngine retrieve_relevant_documents query token error')
        vectorized_query = self._vectorizer.vectorize(tokenized_query)
        if not isinstance(vectorized_query,tuple):
            raise ValueError('VectorDBSearchEngine retrieve_relevant_documents query vector error')
        vectors_list = [pair[1] for pair in self._db.get_vectors()]
        neighbours = self._calculate_knn(vectorized_query,vectors_list,n_neighbours)
        if not neighbours:
            raise ValueError('VectorDBSearchEngine retrieve_relevant_documents neighbours error')
        needed_docs = self._db.get_raw_documents(tuple(pair[0] for pair in neighbours))
        return [(pair[-1], needed_docs[index]) for index, pair in enumerate(neighbours)]


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
        if not isinstance(new_centroid,tuple) or not new_centroid:
            raise ValueError('ClusterDT0 set_new_centroid input error')
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
        if not isinstance(index,int) or index is None or index < 0:
            raise ValueError('ClusterDT0 add_document_index input error')
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
        start_centroids = self._db.get_vectors()[:self._n_clusters]
        self.__clusters = [ClusterDTO(pair[1]) for pair in start_centroids]
        while True:
            new_clusters = self.run_single_train_iteration()
            if self._is_convergence_reached(new_clusters):
                break
        self.__clusters = new_clusters

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        used_centroids = []
        for cluster in self.__clusters:
            cluster.erase_indices()
            used_centroid = cluster.get_centroid()
            used_centroids.append(used_centroid)
        used_vectors = self._db.get_vectors()
        for vector in used_vectors:
            vector_distances = []
            for centroid in used_centroids:
                distance = calculate_distance(vector[1],centroid)
                if distance is None:
                    raise ValueError('Kmeans run_single_train_iteration vector distance error')
                vector_distances.append((distance,used_centroids.index(centroid)))
            close = min(vector_distances)
            self.__clusters[close[1]].add_document_index(used_vectors.index(vector))
        for cluster in self.__clusters:
            cluster_vectors = [used_vectors[index][1] for index in cluster.get_indices()]
            updated_centroid = [sum(vec[index] for index in range(len(vec))) / len(cluster_vectors)
                                for vec in cluster_vectors]
            cluster.set_new_centroid(tuple(updated_centroid))
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
        if not isinstance(query_vector,tuple) or not query_vector\
                or not isinstance(n_neighbours,int) or not n_neighbours:
            raise ValueError("KMeans infer input error")
        distances = []
        for cluster in self.__clusters:
            used_centroid = cluster.get_centroid()
            if not used_centroid:
                continue
            distance = calculate_distance(query_vector,used_centroid)
            if distance is None:
                raise ValueError('KMeans infer query_vector distance error')
            distances.append(distance)
        closest = distances.index(min(distances))
        used_cluster = self.__clusters[closest]
        indices = used_cluster.get_indices()
        used_vectors = self._db.get_vectors(indices)
        new_distances = []
        for index, vector in used_vectors:
            distance = calculate_distance(query_vector,vector)
            if distance is None:
                raise ValueError('KMeans infer new query_vector distance error')
            new_distances.append((distance,index))
        return sorted(new_distances,key=lambda x:x[0])[:n_neighbours]

    def get_clusters_info(self, num_examples: int) -> list[dict[str, int | list[str]]]:
        """
        Get clusters information.

        Args:
            num_examples (int): Number of examples for each cluster

        Returns:
            list[dict[str, int| list[str]]]: List with information about each cluster
        """
        if not isinstance(num_examples, int) or num_examples <=0:
            raise ValueError('KMeans get_clusters_info input error')
        info = []
        for index, cluster in enumerate(self.__clusters):
            used_centroid = cluster.get_centroid()
            used_indices = cluster.get_indices()
            used_vectors = [self._db.get_vectors()[ind] for ind in used_indices]
            distances = []
            for vector in used_vectors:
                distance = calculate_distance(used_centroid,vector[-1])
                if distance is None:
                    raise ValueError('KMeans get_cluster_info distance is None')
                distances.append((distance,vector[0]))
            distances.sort(key=lambda x:x[-1])
            needed_distances = distances[:num_examples]
            new_indices = tuple(pair[-1] for pair in needed_distances)
            used_documents = self._db.get_raw_documents(new_indices)
            single_info = {}
            if isinstance(index,int) and isinstance(used_documents,list):
                single_info.update(cluster_id=index,documents=used_documents)
            info.append(single_info)
        return info

    def calculate_square_sum(self) -> float:
        """
        Get sum of squares of distance from vectors of clusters to their centroid.

        Returns:
            float: Sum of squares of distance from vector of clusters to centroid.
        """
        square_sum = 0.0
        used_vectors = self._db.get_vectors()
        for cluster in self.__clusters:
            used_centroid = cluster.get_centroid()
            used_indices = cluster.get_indices()
            cluster_vecs = [used_vectors[i][1] for i in used_indices]
            square_sum += sum(sum((used_centroid[i]) ** 2 for i
                                in range(len(used_centroid))) for vec in cluster_vecs)
        return square_sum

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
        if not isinstance(new_clusters,list) or not new_clusters or not\
                isinstance(threshold,float) or not threshold:
            raise ValueError('KMeans _is_convergence_reached input error')
        for index,old in enumerate(self.__clusters):
            distance = calculate_distance(old.get_centroid(),new_clusters[index].get_centroid())
            if not isinstance(distance,float):
                raise ValueError('KMeans _is_convergence_reached distance is float')
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
        self.__algo = KMeans(db,n_clusters)
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
        if not query or not isinstance(query,str) or not n_neighbours\
                or not isinstance(n_neighbours,int):
            raise ValueError('ClusteringSE retrieve_relevant_documents input error')
        tokenized_query = self._db.get_tokenizer().tokenize(query)
        if tokenized_query is None:
            raise ValueError('ClusteringSE retrieve_relevant_documents tokenized_query is None')
        query_vector = self._db.get_vectorizer().vectorize(tokenized_query)
        if query_vector is None:
            raise ValueError('ClusteringSE retrieve_relevant_documents query_vector is None')
        neighbours = self.__algo.infer(query_vector,n_neighbours)
        if neighbours is None:
            raise ValueError('ClusteringSearchEngine retrieve_relevant_documents no neighbours')
        indices = tuple(pair[-1] for pair in neighbours)
        documents = self._db.get_raw_documents(indices)
        return [(doc[0],documents[index]) for index,doc in enumerate(neighbours)]


    def make_report(self, num_examples: int, output_path: str) -> None:
        """
        Create report by clusters.

        Args:
            num_examples (int): number of examples for each cluster
            output_path (str): path to output file
        """
        with open(output_path, 'w', encoding='utf-8') as file:
            info = self.__algo.get_clusters_info(num_examples)
            json.dump(info,file)

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
        super().__init__(db, SearchEngine(db.get_vectorizer(),db.get_tokenizer()))
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
