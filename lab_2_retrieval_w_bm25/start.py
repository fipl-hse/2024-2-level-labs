"""
Laboratory Work #2 starter
"""
# pylint:disable=too-many-locals, unused-argument, unused-variable, too-many-branches, too-many-statements


def main() -> None:
    """
    Launches an implementation
    """
    with open("assets/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().split("\n")
    result = None
    assert result, "Result is None"


if __name__ == "__main__":
    main()
