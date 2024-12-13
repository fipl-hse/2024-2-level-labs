"""
Lab 4.

Vector search with clusterization
"""
from astroid import Raise

# pylint: disable=undefined-variable, too-few-public-methods, unused-argument, duplicate-code, unused-private-member, super-init-not-called

from lab_2_retrieval_w_bm25.main import calculate_bm25
from lab_3_ann_retriever.main import BasicSearchEngine, Tokenizer, Vector, Vectorizer, calculate_distance

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
        raise ValueError('Unacceptable argument type or no argument')
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
        super().__init__(corpus=[])
        self._avg_doc_len = -1.0
        self._corpus = []

    def set_tokenized_corpus(self, tokenized_corpus: TokenizedCorpus) -> None:
        """
        Set tokenized corpus and average document length.

        Args:
            tokenized_corpus (TokenizedCorpus): Tokenized texts corpus.

        Raises:
            ValueError: In case of inappropriate type input argument or if input argument is empty.
        """
        if not isinstance(tokenized_corpus, list) or not tokenized_corpus:
            raise ValueError('Unacceptable argument type or no argument')

        self._corpus = tokenized_corpus
        self._avg_doc_len = sum([len(text) for text
                                 in self._corpus]) / len(self._corpus)

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
        if (not isinstance(tokenized_document, list)
                or not all(isinstance(item, str)
                           for item in tokenized_document)
                or not tokenized_document or tokenized_document is None):
            raise ValueError('Unacceptable argument type or no argument')

        res = self._calculate_bm25(tokenized_document)

        if res is None:
            raise ValueError('Result is None')
        return res


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
        if (not isinstance(tokenized_document, list)
                or not all(isinstance(item, str)
                           for item in tokenized_document) or not tokenized_document):
            raise ValueError('Unacceptable argument type or no argument')

        empty_list = [0.0] * len(self._vocabulary)
        res = calculate_bm25(self._vocabulary, tokenized_document,
                             self._idf_values, avg_doc_len=self._avg_doc_len,
                             doc_len=len(tokenized_document))
        for val, ind in self._token2ind.items():
            empty_list[ind] = res[val]

        return tuple(empty_list)





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
        if not isinstance(corpus, list) or not corpus:
            raise ValueError('No argument or unacceptable type of argument')

        tok_pars = []

        for item in corpus:
            tok_text = self._tokenizer.tokenize(item)
            if tok_text is None or not isinstance(tok_text, list):
                raise ValueError('Tokenized text is None or not isinstance list')

            if len(tok_text):
                tok_pars.append(tok_text)
                self.__documents.append(item)

        self._vectorizer.set_tokenized_corpus(tok_pars)
        self._vectorizer.build()

        vect_pars = []

        for text in tok_pars:
            vector = self._vectorizer.vectorize(text)
            if not isinstance(vector, tuple) or not vector:
                raise ValueError('No vector or vector not isinstance tuple')

            vect_pars.append(vector)

        self.__vectors = {ind: vector for ind, vector in enumerate(vect_pars)}

        if None in self.__vectors.values():
            raise ValueError('Vector(-s) is None')

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
        res = []

        if indices is None:
            for key, value in self.__vectors.items():
                res.append((key, value))
            return res

        for key, value in self.__vectors.items():
            if key not in indices:
                continue
            res.append((key, value))

            return res


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
        if not isinstance(indices, tuple) and indices is not None:
            raise ValueError('Not argument or argument is None or unacceptable argument type')

        if indices is None:
            return self.__documents

        documents = []
        new_ind = []

        for ind in indices:
            if not ind in new_ind:
                new_ind.append(ind)

        for ind in new_ind:
            documents.append(self.__documents[ind])
        return documents



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
        if (not isinstance(query, str)
                or query is None or not query or not isinstance(n_neighbours, int)):
            raise ValueError('Unacceptable arguments/types')

        if n_neighbours < 0:
            raise ValueError('There is no quantity of neighbours')

        tok_query = self._tokenizer.tokenize(query)
        if tok_query is None or not tok_query:
            raise ValueError('Problem with tokenization')

        vect_query = self._vectorizer.vectorize(tok_query)
        if vect_query is None or not vect_query:
            raise ValueError('Problem with vectorization')

        inf = [elem[1] for elem in self._db.get_vectors()]
        res = self._calculate_knn(vect_query, inf, n_neighbours)

        if res is None or not res:
            raise ValueError('Problem with calculating KNN')

        ind = tuple([doc_inf[0] for doc_inf in res])

        documents = self._db.get_raw_documents(ind)

        result = [(doc_inf[1], documents[ind]) for ind, doc_inf in enumerate(res)]
        return result



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
            raise ValueError('Empty argument or argument is not tuple')

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
        if not isinstance(index, int) or not index or index < 0:
            raise ValueError('Empty argument or argument is not int or argument < 0')

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
        first_centroids = self._db.get_vectors()[:self._n_clusters]
        for item in first_centroids:
            self.__clusters.append(ClusterDTO(item[1]))
        self.run_single_train_iteration()

        while self._is_convergence_reached(self.__clusters) is False:
            self.run_single_train_iteration()

    def run_single_train_iteration(self) -> list[ClusterDTO]:
        """
        Run single train iteration.

        Raises:
            ValueError: In case of if methods used return None.

        Returns:
            list[ClusterDTO]: List of clusters.
        """
        centroids = []
        for item in self.__clusters:
            item.erase_indices()
            element = item.get_centroid()
            centroids.append(element)
        vects = self._db.get_vectors()

        for ind, vect in vects:
            dist_list = []
            for item in centroids:
                dist = calculate_distance(vect, item)
                if dist is None:
                    raise ValueError('calculate_distance returns None')
                dist_list.append(dist)
            closest = min(dist_list)
            self.__clusters[dist_list.index(closest)].add_document_index(ind)

        for elem in self.__clusters:
            vectors = [vects[index][-1] for index in elem.get_indices()]
            new_centr = [sum(element[item1] for item1
                             in range(len(element))) / len(vectors) for element in vectors]
            elem.set_new_centroid(tuple(new_centr))
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
        if not query_vector or not isinstance(query_vector, tuple) or n_neighbours < 0:
            raise ValueError('Problem(-s) with arguments')
        if query_vector is None:
            raise ValueError('Query is None')

        dist_to_cluster = []
        for cluster in self.__clusters:
            if cluster.get_centroid() is None:
                continue
            dist = calculate_distance(query_vector, cluster.get_centroid())
            if dist is None:
                raise ValueError('1-st distance is None')
            dist_to_cluster.append(dist)

        close_cluster = self.__clusters[dist_to_cluster.index(min(dist_to_cluster))]

        if close_cluster is None:
            raise ValueError('No closest cluster')

        if close_cluster.get_indices() is None:
            raise ValueError('Get indices return None')

        res = []
        for ind, val in self._db.get_vectors(close_cluster.get_indices()):
            final_dist = calculate_distance(query_vector, val)
            if final_dist is None:
                raise ValueError('Distance between query and vector in None')
            res.append((final_dist, ind))

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
        if (not isinstance(new_clusters, list) or
                not new_clusters or not isinstance(threshold, float)
                or not threshold or new_clusters is None or threshold is None):
            raise ValueError('Unacceptable argument(-s)/their type(-s)/None returns')

        if not self.__clusters:
            raise ValueError('Clusters return None or empty clusters')

        for ind, elem in enumerate(new_clusters):
            old = self.__clusters[ind].get_centroid()
            new = elem.get_centroid()
            dist = calculate_distance(old, new)
            if dist is None or not dist:
                raise ValueError('Distance error between previous and new centroids')
            return True if dist < threshold else False


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
        if not query or not isinstance(query, str) or query is None or not isinstance(n_neighbours, int):
            raise ValueError('Problem(-s) with arguments')

        tok_q = self._db.get_tokenizer().tokenize(query)
        if tok_q is None:
            raise ValueError('tokenized query is None')
        vect_q = self._db.get_vectorizer().vectorize(tok_q)
        if vect_q is None:
            raise ValueError('vectorized query is None')

        self.__algo.train()

        inf = self.__algo.infer(vect_q, n_neighbours)
        if inf is None:
            raise ValueError('infer result is None')
        indexes = [ind[1] for ind in inf]
        relevant_neighbours = self._db.get_raw_documents(tuple(indexes))
        return [(inf[i][0], relevant_neighbours[i]) for i in range(len(inf))]

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
