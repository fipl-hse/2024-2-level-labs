"""
Run start.
"""
import argparse
import subprocess
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.collect_coverage.run_coverage import get_target_score
from config.constants import CONFIG_PACKAGE_PATH, PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.project_config import ProjectConfig


def check_skip_conditions(pr_name: str, repository_type: str) -> bool:
    """
    Check skip conditions based on repository type and admin status.

    Args:
        pr_name (str): Pull request name.
        repository_type (str): Type of the repository.

    Returns:
        bool: True if skip conditions are met, otherwise False.
    """
    result = _run_console_tool(
        str(choose_python_exe()),
        [str(Path(CONFIG_PACKAGE_PATH, 'is_admin.py')), '--pr_name', pr_name],
        debug=True
    )
    if repository_type == "public" and result.stdout.decode("utf-8").strip() == 'YES':
        print('[skip-lab] option was enabled, skipping check...')
        return True
    return False


@handles_console_error()
def run_start(lab_name: str) -> subprocess.CompletedProcess:
    """
    Run start.py script in the specified lab directory.

    Args:
        lab_name (str): Name of the lab directory.

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    return _run_console_tool(
        str(choose_python_exe()),
        [str('start.py')],
        cwd=PROJECT_ROOT / lab_name,
        debug=True
    )


@handles_console_error()
def check_start_content(lab_name: str) -> subprocess.CompletedProcess:
    """
    Check the content of start.py script using check_start_content.py script.

    Args:
        lab_name (str): Name of the lab directory.

    Returns:
        subprocess.CompletedProcess: Program execution values.
    """
    lab_dir = Path(lab_name)
    start_py_file = lab_dir / 'start.py'
    with start_py_file.open() as f:
        start_py_content = f.read()

    return _run_console_tool(str(choose_python_exe()),
                             [str(Path(CONFIG_PACKAGE_PATH, 'check_start_content.py')),
                              '--start_py_content', start_py_content],
                             cwd=PROJECT_ROOT,
                             debug=True)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run start.py checks for each lab.")
    parser.add_argument("--pr_name", help="Pull request name")
    parser.add_argument("--repository_type", help="Type of the repository (public/private)")
    return parser.parse_args()


def main() -> None:
    """
    Main function to run start.py checks for each lab.
    """
    args = parse_arguments()
    pr_name = args.pr_name
    repository_type = args.repository_type

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs = project_config.get_labs_names()

    for lab_name in labs:
        print(f"Running start.py checks for lab {lab_name}")

        if check_skip_conditions(pr_name, repository_type):
            continue

        target_score = get_target_score(PROJECT_ROOT / lab_name)

        if target_score == 0:
            print("Skipping stage")
            continue
        run_start(lab_name)

        print(f"Check calling lab {lab_name} passed")

        check_start_content(lab_name)

    print("All start.py checks passed.")


if __name__ == "__main__":
    main()
