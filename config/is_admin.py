"""
Check whether PR author is admin or not.
"""

from tap import Tap


class ArgumentParser(Tap):
    """
    CLI for script.
    """

    pr_name: str


def main() -> None:
    """
    Main module entrypoint.
    """
    args = ArgumentParser().parse_args()

    if "[skip-lab]" in args.pr_name:
        print("YES")
    else:
        print("NO")


if __name__ == "__main__":
    main()
