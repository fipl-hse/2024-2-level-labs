"""
Run tests for each lab using pytest.
"""
import subprocess
import sys

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.common import check_result
from config.constants import PROJECT_CONFIG_PATH
from config.project_config import ProjectConfig


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
    pr_name = sys.argv[1]
    pr_author = sys.argv[2]

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs = project_config.get_labs_names()

    print(f"Current scope: {labs}")

    for lab_name in labs:
        print(f"Running tests for lab {lab_name}")

        result = run_tests(pr_name, pr_author, lab_name)

        print(result.stdout.decode('utf-8'))
        print(result.stderr.decode('utf-8'))

        check_result(result.returncode)

    print("All tests passed.")


if __name__ == "__main__":
    main()
