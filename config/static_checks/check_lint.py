"""
Check lint for code style in Python code.
"""
# pylint: disable=duplicate-code
import argparse
from os import listdir
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.lab_settings import LabSettings
from config.project_config import ProjectConfig


@handles_console_error()
def check_lint_on_paths(
        paths: list[Path], path_to_config: Path,
        exit_zero: bool = False,
        ignore_tests: bool = False
) -> tuple[str, str, int]:
    """
    Run lint checks for the project.

    Args:
        paths (list[Path]): Paths to the projects.
        path_to_config (Path): Path to the config.
        exit_zero (bool): Exit-zero lint argument.
        ignore_tests (bool): Ignore lint argument.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    lint_args = [
        "-m",
        "pylint",
        *map(str, filter(lambda x: x.exists(), paths))
        ,
        "--rcfile",
        str(path_to_config)
    ]
    if ignore_tests:
        lint_args.extend(["--ignore", "tests"])
    if exit_zero:
        lint_args.append("--exit-zero")
    return _run_console_tool(str(choose_python_exe()), lint_args,
                             debug=True)


@handles_console_error()
def check_lint_level(lint_output: str, target_score: int) -> tuple[str, str, int]:
    """
    Run lint level check for the project.

    Args:
        lint_output (str): Pylint check output.
        target_score (int): Target score.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    lint_level_args = [
        "-m",
        "config.static_checks.lint_level",
        "--lint-output",
        lint_output,
        "--target-score",
        str(target_score)
    ]

    return _run_console_tool(str(choose_python_exe()), lint_level_args, debug=True)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run check_lint.py checks for each lab.")
    parser.add_argument("--repository_type", help="Type of the repository (public/private)")
    return parser.parse_args()


def main() -> None:
    """
    Run lint checks for the project.
    """
    args = parse_arguments()
    repository_type = args.repository_type
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths()

    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    print("Running lint on config, seminars, admin_utils")
    stdout, _, _ = check_lint_on_paths(
        [
            PROJECT_ROOT / "config",
            PROJECT_ROOT / "seminars",
            PROJECT_ROOT / "admin_utils"
        ],
        pyproject_path, exit_zero=True)
    check_lint_level(stdout, 10)

    if (PROJECT_ROOT / "core_utils").exists():
        print("core_utils exist")
        print("Running lint on core_utils")
        stdout, _, _ = check_lint_on_paths(
            [
                PROJECT_ROOT / "core_utils"
            ],
            pyproject_path, exit_zero=True)
        check_lint_level(stdout, 10)

    for lab_name in labs_list:
        lab_path = PROJECT_ROOT / lab_name

        if "settings.json" in listdir(lab_path):
            target_score = LabSettings(PROJECT_ROOT / f"{lab_path}/settings.json").target_score
            if target_score == 0:
                print("Skipping check")
                continue

            print(f"Running lint for lab {lab_path}")
            stdout, _, _ = check_lint_on_paths(
                [
                    lab_path
                ],
                pyproject_path, ignore_tests=repository_type == "public", exit_zero=True)
            check_lint_level(stdout, target_score)


if __name__ == "__main__":
    main()
