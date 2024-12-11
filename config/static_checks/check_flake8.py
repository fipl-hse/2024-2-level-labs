"""
Check flake8 to check the style and quality of Python code.
"""

# pylint: disable=duplicate-code
from os import listdir
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.console_logging import get_child_logger
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.lab_settings import LabSettings
from config.project_config import ProjectConfig

logger = get_child_logger(__file__)


@handles_console_error()
def check_flake8_on_paths(paths: list[Path]) -> tuple[str, str, int]:
    """
    Run flake8 checks for the project.

    Args:
        paths (list[Path]): Paths to the projects.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    flake_args = ["-m", "flake8", *map(str, filter(lambda x: x.exists(), paths))]

    return _run_console_tool(str(choose_python_exe()), flake_args, debug=True, cwd=PROJECT_ROOT)


def main() -> None:
    """
    Run flake8 checks for the project.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths()

    logger.info("Running flake8 on config, seminars, admin_utils")
    check_flake8_on_paths(
        [PROJECT_ROOT / "config", PROJECT_ROOT / "seminars", PROJECT_ROOT / "admin_utils"]
    )

    if (PROJECT_ROOT / "core_utils").exists():
        logger.info("core_utils exist")
        check_flake8_on_paths([PROJECT_ROOT / "core_utils"])

    for lab_name in labs_list:
        lab_path = PROJECT_ROOT / lab_name
        if "settings.json" in listdir(lab_path):
            target_score = LabSettings(PROJECT_ROOT / f"{lab_path}/settings.json").target_score

            if target_score > 5:
                logger.info(f"Running flake8 for lab {lab_path}")
                check_flake8_on_paths([lab_path])


if __name__ == "__main__":
    main()
