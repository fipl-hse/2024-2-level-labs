"""
Run tests for each lab using pytest.
"""
import subprocess

from tap import Tap

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.common import check_result
from config.constants import PROJECT_CONFIG_PATH
from config.project_config import ProjectConfig


class CommandLineInterface(Tap):
    """
    Types for the argument parser.
    """
    pr_name: str
    pr_author: str


def run_tests(pr_name: str, pr_author: str, lab: str) -> subprocess.CompletedProcess:
    """
    Run pytest tests for a specific lab.

    Args:
        pr_name (str): Pull request name.
        pr_author (str): Pull request author.
        lab (str): Name of the lab.

    Returns:
        subprocess.CompletedProcess: Result of the subprocess execution.
    """
    args = [
        str('config/run_pytest.py'),
        '--pr_name', pr_name,
        '--pr_author', pr_author,
        '-l', lab,
        '-m', lab
    ]
    return _run_console_tool(str(choose_python_exe()), args, debug=True)


def main() -> None:
    """
    Main function to run tests.
    """
    args = CommandLineInterface(underscores_to_dashes=True).parse_args()

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs = project_config.get_labs_names()

    print(f"Current scope: {labs}")

    for lab_name in labs:
        print(f"Running tests for lab {lab_name}")

        result = run_tests(args.pr_name, args.pr_author, lab_name)

        print(result.stdout.decode('utf-8'))
        print(result.stderr.decode('utf-8'))

        check_result(result.returncode)

    print("All tests passed.")


if __name__ == "__main__":
    main()
