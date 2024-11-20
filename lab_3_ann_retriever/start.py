"""
Laboratory Work #3 starter.
"""

# pylint:disable=duplicate-code, too-many-locals, too-many-statements, unused-variable
from pathlib import Path
from lab_3_ann_retriever.main import Tokenizer, Vectorizer, BasicSearchEngine


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
    with open("assets/secrets/secret_1.txt", "r", encoding="utf-8") as text_file:
        text = text_file.read()

    tokenizer = Tokenizer(open_files()[1])
    result = tokenizer.tokenize_documents(open_files()[0])
    assert result, "Result is None"

    tokenizer = Tokenizer(open_files()[0])
    tok_document = tokenizer.tokenize_documents(open_files()[0])
    vectorizer = Vectorizer(tok_document)
    vectorizer.build()
    secret_vector = tuple(float(item) for item in text.split(', '))
    print(vectorizer.vector2tokens(secret_vector))
    basic_search = BasicSearchEngine(vectorizer, tokenizer)
    print(basic_search.retrieve_vectorized(secret_vector))

if __name__ == "__main__":
    main()
