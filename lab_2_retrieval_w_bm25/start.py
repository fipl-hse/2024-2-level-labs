"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
from lab_2_retrieval_w_bm25.main import (build_vocabulary, calculate_tf, calculate_idf,
                                         calculate_tf_idf, tokenize, remove_stopwords)


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
    text_1 = remove_stopwords(tokenize('''There was a boy, who was a wizard. He used his wand to do
    spells. He was studying in a magic school. His best friend was a wizard too.'''), stopwords)
    text_2 = remove_stopwords(tokenize('''Steven was a boy who loved pets. He had a cat, two dogs
    and three parrots. Every morning he did not want to go to school, because he had to leave his
    pets at home.'''), stopwords)
    text_3 = remove_stopwords(tokenize('''A dragon and a princess had a picnic date at the top
    of the hill. They rarely leaved the tower, but the summer weather was just perfect for the
    hill picnic.'''), stopwords)
    tf = calculate_tf(build_vocabulary([text_1, text_2, text_3]), text_1)
    idf = calculate_idf(build_vocabulary([text_1, text_2, text_3]), [text_1, text_2, text_3])
    result = calculate_tf_idf(tf, idf)
    print(result)
    assert result, "Result is None"


if __name__ == "__main__":
    main()
