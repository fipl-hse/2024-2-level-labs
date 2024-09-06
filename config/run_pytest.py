"""
Run pytest for a specific lab.
"""
import argparse
import subprocess
import sys

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.collect_coverage.run_coverage import get_target_score
from config.common import check_skip
from config.constants import PROJECT_ROOT


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run pytest for a specific lab with given labels.")
    parser.add_argument("--pr_name", help="Pull request name")
    parser.add_argument("--pr_author", help="Pull request author")
    parser.add_argument("-l", "--lab-path", help="Path to the lab")
    parser.add_argument("-m", "--pytest-label", help="Label for pytest")
    return parser.parse_args()


def prepare_pytest_args(lab_path: str, target_score: int, pytest_label: str) -> list[str]:
    """
    Build the arguments for running pytest.

    Args:
        lab_path (str): Path to the lab.
        target_score (int): Target score for the lab.
        pytest_label (str): Label for pytest.

    Returns:
        list[str]: List of arguments for pytest.
    """
    if lab_path:
        label = f"mark{target_score} and {pytest_label}"
    else:
        label = pytest_label

    if lab_path == "lab_5_scrapper":
        pytest_args = ["-m", label, "--capture=no", "--ignore=lab_6_pipeline"]
    elif lab_path == "lab_6_pipeline" and target_score == 10:
        pytest_args = ["-m", label, "--capture=no"]
    else:
        pytest_args = ["-m", label, "--capture=no",
                       "--ignore=lab_6_pipeline/tests/s4_pos_frequency_pipeline_test.py",
                       "--ignore=lab_6_pipeline/tests/s3_6_advanced_pipeline_test.py"]
    return pytest_args


def run_pytest(pytest_args: list[str]) -> subprocess.CompletedProcess:
    """
    Run pytest with the given arguments.

    Args:
        pytest_args (list[str]): Arguments for pytest.

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    args = ["-m", "pytest"] + pytest_args
    return _run_console_tool(
        str(choose_python_exe()),
        args,
        cwd=PROJECT_ROOT,
        debug=True
    )


def main() -> None:
    """
    Main function to run tests for each lab.
    """
    args = parse_arguments()
    pr_name = args.pr_name
    pr_author = args.pr_author
    lab_path = args.lab_path
    pytest_label = args.pytest_label

    if lab_path:
        check_skip(pr_name, pr_author, lab_path)
        score_path = PROJECT_ROOT / lab_path
        target_score = get_target_score(score_path)
    else:
        target_score = None

    pytest_args = prepare_pytest_args(lab_path, target_score, pytest_label)

    ret = run_pytest(pytest_args)

    print(ret.stdout.decode("utf-8"))
    print(ret.stderr.decode("utf-8"))

    if ret.returncode == 5:
        print("No tests collected. Exiting with 0 (instead of 5).")
        sys.exit(0)

    print(f"Pytest results (should be 0): {ret.returncode}")
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
