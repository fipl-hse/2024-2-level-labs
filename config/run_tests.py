"""
Run tests for each lab using pytest.
"""

from typing import Optional

from tap import Tap

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.collect_coverage.run_coverage import get_target_score
from config.common import check_skip
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig


class CommandLineInterface(Tap):
    """
    Types for the argument parser.
    """

    pr_name: str
    pr_author: str
    lab_path: Optional[str] = None
    pytest_label: Optional[str] = None


def prepare_pytest_args(
    lab_path: str, target_score: int, pytest_label: Optional[str] = None
) -> list[str]:
    """
    Build the arguments for running pytest.

    Args:
        lab_path (str): Path to the lab.
        target_score (int): Target score for the lab.
        pytest_label (Optional[str]): Label for pytest.

    Returns:
        list[str]: List of arguments for pytest.
    """
    if pytest_label is None:
        pytest_label = lab_path

    pytest_args = [
        "-m",
        f"mark{target_score} and {pytest_label}" if lab_path else pytest_label,
        "--capture=no",
    ]
    print(pytest_args)

    if lab_path == "lab_5_scrapper":
        pytest_args.append("--ignore=lab_6_pipeline")

    return pytest_args


@handles_console_error()
def run_pytest(pytest_args: list[str]) -> tuple[str, str, int]:
    """
    Run pytest with the given arguments.

    Args:
        pytest_args (list[str]): Arguments for pytest.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    args = ["-m", "pytest", *pytest_args]
    return _run_console_tool(str(choose_python_exe()), args, cwd=PROJECT_ROOT, debug=True)


def main() -> None:
    """
    Main function to run tests for only one lab or for one by one.
    """
    args = CommandLineInterface(underscores_to_dashes=True).parse_args()

    if args.lab_path:
        if check_skip(args.pr_name, args.lab_path):
            return
        target_score = get_target_score(PROJECT_ROOT / args.lab_path)
        pytest_args = prepare_pytest_args(args.lab_path, target_score, args.pytest_label)

        run_pytest(pytest_args)

    else:
        project_config = ProjectConfig(PROJECT_CONFIG_PATH)
        labs = project_config.get_labs_names()

        print(f"Current scope: {labs}")

        for lab_name in labs:
            if check_skip(args.pr_name, lab_name):
                continue
            print(f"Running tests for lab {lab_name}")

            target_score = get_target_score(PROJECT_ROOT / lab_name)
            pytest_args = prepare_pytest_args(lab_name, target_score)

            run_pytest(pytest_args)


if __name__ == "__main__":
    main()
