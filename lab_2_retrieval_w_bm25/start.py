"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code

from lab_2_retrieval_w_bm25.main import (remove_stopwords, tokenize)

def main() -> None:
    """
    Launches an implementation
    """
    paths_to_texts = [
        "assets/fairytale_1.txt",
        "assets/fairytale_2.txt",
        "assets/fairytale_3.txt",
        "assets/fairytale_4.txt",
        "assets/fairytale_5.txt",
        "assets/fairytale_6.txt",
        "assets/fairytale_7.txt",
        "assets/fairytale_8.txt",
        "assets/fairytale_9.txt",
        "assets/fairytale_10.txt",
    ]
    documents = []
    for path in paths_to_texts:
        with open(path, "r", encoding="utf-8") as file:
            documents.append(file.read())
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    documents.append(file.read())
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    tokens = []
    clear_tokens = []
    for doc in documents:
        tok_doc = tokenize(doc)
        tokens.append(tok_doc)
        if tok_doc is not None:
            clear_tok_doc = remove_stopwords(tok_doc, stopwords)
            if clear_tok_doc is not None:
                clear_tokens.append(clear_tok_doc)
    result = documents
    assert result, "Result is None"
    return None


if __name__ == "__main__":
    main()
