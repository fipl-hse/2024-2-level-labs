"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
from lab_3_ann_retriever.main import Tokenizer, Vectorizer, BasicSearchEngine, Node, NaiveKDTree, SearchEngine, KDTree


def open_files() -> tuple[list[str], list[str]]:
    """
    # stubs: keep.

    Open files.

    Returns:
        tuple[list[str], list[str]]: Documents and stopwords
    """
    documents = []
    for path in sorted(Path("assets/articles").glob("*.txt")):
        with open(path, "r", encoding="utf-8") as file:
            documents.append(file.read())
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    return (documents, stopwords)


def main() -> None:
    """
    Launch an implementation.
    """
    with open("assets/secrets/secret_5.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()
    secret_vector = tuple(float(item) for item in text.split(', '))
    result = ':('
    query = 'русскому языку ЕГэ 11 классов'
    stopwords = open_files()[1]
    tokenizer = Tokenizer(stopwords)
    # documents = open_files()[0]
    documents = ['Векторы используются для поиска релевантного документа. Давайте научимся, как их создавать и использовать!', 'Мой кот Вектор по утрам приносит мне тапочки, а по вечерам мы гуляем с ним на шлейке во дворе. Вектор забавный и храбрый. Он не боится собак!', 'Котёнок, которого мы нашли во дворе, очень забавный и пушистый. По утрам я играю с ним в догонялки перед работой.', 'Моя собака думает, что её любимый плед — это кошка. Просто он очень пушистый и мягкий. Забавно наблюдать, как они спят вместе!']
    tokenized_documents = tokenizer.tokenize_documents(documents)
    vectorizer = Vectorizer(tokenized_documents)
    vectorizer.build()
    knn_retriever = BasicSearchEngine(vectorizer=vectorizer, tokenizer=tokenizer)
    knn_retriever.index_documents(documents)
    print(knn_retriever._document_vectors)
    # print(knn_retriever.retrieve_relevant_documents("Мои кот и собака не дружат!", 2))
    # print(vectorizer.vector2tokens(secret_vector))
    # # print(knn_retriever.retrieve_vectorized(secret_vector))
    # node = Node((0.0, 0.0), -1, None, Node((0.1, 0.1), 0))
    # naive_tree = NaiveKDTree()
    # vectors = knn_retriever._document_vectors
    # naive_tree.build(vectors)
    # query_vector = vectorizer.vectorize(tokenizer.tokenize(query))
    # naive_tree.query(query_vector)
    # naive_kdtree_retriever = SearchEngine(vectorizer, tokenizer)
    # naive_kdtree_retriever.index_documents(documents)
    # print(naive_kdtree_retriever.retrieve_relevant_documents(query))
    # print(knn_retriever.retrieve_relevant_documents(query, 3))
    # tree = KDTree()
    # vs = [(0.0, 0.0, 0.094), (0.061, 0.121, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    # tree.build(vs)
    # query_vector = (0.0, 0.0, 0.094)
    # print(tree.query(query_vector, 4))
    assert result, "Result is None"


if __name__ == "__main__":
    main()
