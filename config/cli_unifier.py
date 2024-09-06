"""
CLI commands.
"""

import platform
import subprocess
from pathlib import Path
from typing import Any


def choose_python_exe() -> Path:
    """
    Select python binary path depending on current OS.

    Returns:
        Path: A path to python exe
    """
    lab_path = Path(__file__).parent.parent
    if platform.system() == 'Windows':
        python_exe_path = lab_path / 'venv' / 'Scripts' / 'python.exe'
    else:
        python_exe_path = lab_path / 'venv' / 'bin' / 'python'
    return python_exe_path


def prepare_args_for_shell(args: list[object]) -> str:
    """
    Prepare argument for CLI.

    Args:
        args (list[object]): arguments to join

    Returns:
        str: arguments for CLI
    """
    return " ".join(map(str, args))


def _run_console_tool(exe: str, /, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """
    Run CLI commands.

    Args:
        exe (str): A path to python exe
        args (list[str]): Arguments
        **kwargs (Any): Options

    Returns:
        subprocess.CompletedProcess: Program execution values
    """
    kwargs_processed: list[str] = []
    for item in kwargs.items():
        if item[0] in ('env', 'debug', 'cwd'):
            continue
        kwargs_processed.extend(map(str, item))

    options = [
        str(exe),
        *args,
        *kwargs_processed
    ]

    if kwargs.get('debug', False):
        print(f'Attempting to run with the following arguments: {" ".join(options)}')

    env = kwargs.get('env')
    if env:
        # pylint:disable = subprocess-run-check
        return subprocess.run(options, capture_output=True, env=env)
    if kwargs.get('cwd'):
        # pylint:disable = subprocess-run-check
        return subprocess.run(options, capture_output=True, cwd=kwargs.get('cwd'))
    # pylint:disable = subprocess-run-check
    return subprocess.run(options, capture_output=True)
