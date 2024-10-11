"""
Common functions for checks.
"""
import sys

from config.collect_coverage.run_coverage import get_target_score
from config.constants import PROJECT_ROOT


def check_result(return_code: int) -> None:
    """
    Check result and exit if failed.

    Args:
        return_code (int): Return code of check
    """
    if return_code != 0:
        print("Check failed.")
        sys.exit(1)
    else:
        print("Check passed.")


def check_skip(pr_name: str, lab_path: str) -> None:
    """
    Exit if skip conditions are met.

    Args:
        pr_name (str): Pull request name.
        lab_path (str): Path to the lab.
    """
    if (pr_name is not None) and ('[skip-lab]' in str(pr_name)):
        print('Skipping check due to label.')
        sys.exit(0)

    if lab_path:
        score_path = PROJECT_ROOT / lab_path
        score = get_target_score(lab_path=score_path)
        if score == 0:
            print('Skipping check due to no mark.')
            sys.exit(0)

    print('No special reasons for skip!')
