"""
Common functions for checks.
"""
import sys

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.constants import PROJECT_ROOT


def check_result(return_code: int) -> None:
    """
    Check result and exit if failed.

    Args:
        return_code (int): Return code of check
    """
    print(return_code)
    if return_code != 0:
        print("Check failed.")
        sys.exit(1)
    else:
        print("Check passed.")


def check_skip(pr_name: str, pr_author: str, lab_path: str) -> None:
    """
    Run skip check script and exit if skip conditions are met.

    Args:
        pr_name (str): Pull request name.
        pr_author (str): Pull request author.
        lab_path (str): Path to the lab.
    """
    result = _run_console_tool(
        str(choose_python_exe()),
        [str('config/skip_check.py'), '--pr_name', pr_name,
         '--pr_author', pr_author, '--lab_path', lab_path],
        cwd=PROJECT_ROOT,
        debug=True
    )

    if result.returncode == 0:
        print('skip check due to special conditions...')
        sys.exit(0)

    if result.stderr:
        print(result.stderr.decode('utf-8'))
        raise ValueError('Error running skip_check tool!')
