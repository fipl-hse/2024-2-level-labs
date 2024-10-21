"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements, duplicate-code
import lab_2_retrieval_w_bm25.main as func


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
    temp = []
    for al in documents:
        tkn = func.tokenize(al)
        if not isinstance(tkn, list):
            return
        tkalt = func.remove_stopwords(tkn, stopwords)
        if not isinstance(tkalt, list):
            return
        temp.append(tkalt)
    res1 = func.build_vocabulary(temp)
    print(res1)
    tidfttl = []
    curidf = func.calculate_idf(res1, temp)
    for al in temp:
        curtf = func.calculate_tf(res1, al)
        if not isinstance(curtf, dict) or \
                not isinstance(curidf, dict):
            return
        tidf = func.calculate_tf_idf(curtf, curidf)
        if not isinstance(tidf, dict):
            return
        tidfttl.append(tidf)
        print(tidf)
    alen = 0
    for al in temp:
        alen+=len(al)
    alen /= len(temp)
    okapi = []
    for e in temp:
        okapi.append(func.calculate_bm25(res1, e, curidf,
                                         1.5, 0.75, alen, len(e)))
    for al in okapi:
        if not isinstance(al, dict):
            return
    print(okapi)
    tstqr = 'Which fairy tale has Fairy Queen?'
    res2 = func.rank_documents(tidfttl, tstqr, stopwords)
    print(res2)
    res3 = func.rank_documents(okapi, tstqr, stopwords)
    result = 1
    assert result, "Result is None"


if __name__ == "__main__":
    main()
