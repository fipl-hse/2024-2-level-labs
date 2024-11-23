"""
Common functions for checks.
"""

from config.collect_coverage.run_coverage import get_target_score
from config.constants import PROJECT_ROOT


def check_skip(pr_name: str, lab_path: str) -> bool:
    """
    Exit if skip conditions are met.

    Args:
        pr_name (str): Pull request name.
        lab_path (str): Path to the lab.

    Returns:
        bool: True if should be skipped
    """
    if (pr_name is not None) and ("[skip-lab]" in str(pr_name)):
        print("Skipping check due to label.")
        return True

    if lab_path:
        score_path = PROJECT_ROOT / lab_path
        score = get_target_score(lab_path=score_path)
        if score == 0:
            print(f"Skipping check due to no mark for lab {lab_path}.")
            return True

    print("No special reasons for skip!")
    return False
