"""
Check docstrings for conformance to the Google-style-docstrings.
"""
import subprocess
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe, handles_console_error
from config.constants import (CONFIG_PACKAGE_PATH, CORE_UTILS_PACKAGE_PATH, PROJECT_CONFIG_PATH,
                              PROJECT_ROOT)
from config.project_config import ProjectConfig


def get_files() -> list:
    """
    Get paths to files in config and core_utils packages.

    Returns:
        list: File paths
    """
    return [
        file
        for directory in [CONFIG_PACKAGE_PATH, CORE_UTILS_PACKAGE_PATH]
        for file in directory.glob('**/*.py')
        if file.name != '__init__.py'
    ]

@handles_console_error()
def check_with_pydoctest(path_to_file: Path, path_to_config: Path) \
        -> subprocess.CompletedProcess:
    """
    Check docstrings in file with pydoctest module.

    Args:
        path_to_file (Path): Path to file
        path_to_config (Path): Path to pydoctest config

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    pydoctest_args = [
        '--config',
        str(path_to_config),
        '--file',
        str(path_to_file)
    ]
    return _run_console_tool('pydoctest', pydoctest_args, debug=True)

@handles_console_error()
def check_with_pydocstyle(path_to_file: Path) -> subprocess.CompletedProcess:
    """
    Check docstrings in file with pydocstyle module.

    Args:
        path_to_file (Path): Path to file

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    pydocstyle_args = [
        '-m',
        'pydocstyle',
        str(path_to_file)
    ]
    return _run_console_tool(str(choose_python_exe()), pydocstyle_args,
                             debug=True)


def check_file(path_to_file: Path) -> None:
    """
    Check docstrings in file for conformance to the Google-style-docstrings.

    Args:
        path_to_file (Path): Path to file
    """
    pydoctest_config = PROJECT_ROOT / 'config' / 'static_checks' / 'pydoctest.json'

    check_with_pydoctest(path_to_file, pydoctest_config)

    check_with_pydocstyle(path_to_file)


def main() -> None:
    """
    Check docstrings for lab, config and core_utils packages.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_list = project_config.get_labs_paths()
    files_list = get_files()

    # check docstrings in config and core_utils
    for file in files_list:
        check_file(file)

    # check docstrings in labs
    for lab_path in labs_list:
        paths = (
            lab_path / 'main.py',
            lab_path / 'start.py',
            lab_path / 'scrapper.py',
            lab_path / 'scrapper_dynamic.py',
            lab_path / 'pipeline.py',
        )
        for path in paths:
            if not path.exists():
                print(f'\nIgnoring {path}: it does not exist.')
                continue

            check_file(path)



if __name__ == '__main__':
    main()
