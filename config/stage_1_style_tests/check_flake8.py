"""
Check flake8 to check the style and quality of Python code.
"""
# pylint: disable=duplicate-code
import subprocess
from os import listdir
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.common import check_result
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.lab_settings import LabSettings
from config.project_config import ProjectConfig


def check_flake8_on_paths(paths: list[Path]) -> subprocess.CompletedProcess:
    """
    Run flake8 checks for the project.

    Args:
        paths (list[Path]): Paths to the projects.

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    flake_args = [
        "-m",
        "flake8",
        *map(str, filter(lambda x: x.exists(), paths))
    ]
    return _run_console_tool(str(choose_python_exe()), flake_args, debug=True)


def main() -> None:
    """
    Run flake8 checks for the project.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths()

    print("Running flake8 on config, seminars, admin_utils")
    completed_process = check_flake8_on_paths(
        [
            PROJECT_ROOT / "config",
            PROJECT_ROOT / "seminars",
            PROJECT_ROOT / "admin_utils"
        ]
    )
    print(completed_process.stdout.decode("utf-8"))
    print(completed_process.stderr.decode("utf-8"))
    check_result(completed_process.returncode)

    if (PROJECT_ROOT / "core_utils").exists():
        print("core_utils exist")
        completed_process = check_flake8_on_paths(
            [
                PROJECT_ROOT / "core_utils"
            ]
        )
        print(completed_process.stdout.decode("utf-8"))
        print(completed_process.stderr.decode("utf-8"))
        check_result(completed_process.returncode)

    for lab_name in labs_list:
        lab_path = PROJECT_ROOT / lab_name
        if "settings.json" in listdir(lab_path):
            target_score = LabSettings(
                PROJECT_ROOT / f"{lab_path}/settings.json").target_score

            if target_score > 5:
                print(f"Running flake8 for lab {lab_path}")
                completed_process = check_flake8_on_paths(
                    [
                        lab_path
                    ]
                )
                print(completed_process.stdout.decode("utf-8"))
                print(completed_process.stderr.decode("utf-8"))
                check_result(completed_process.returncode)


if __name__ == "__main__":
    main()
