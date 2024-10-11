"""
CLI commands.
"""
import functools
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

LOG_TEMPLATE = """

>>>>>>>>>>>>>>>>>>>>>>>>> {output_type} >>>>>>>>>>>>>>>>>>>>>>>>>
{content}
<<<<<<<<<<<<<<<<<<<<<<<<< {output_type} <<<<<<<<<<<<<<<<<<<<<<<<<

"""


def log_output(output_type: str, content: bytes) -> None:
    """
    Prints result of the command-line process.

    Args:
        output_type(str): type of output, for example stdout or stderr
        content(bytes): raw result from the subprocess call
    """
    print(
        LOG_TEMPLATE.format(
            output_type=output_type,
            content=content.decode("utf-8").replace('\r', ''),
        )
    )


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


def _run_console_tool(exe: str, /, args: list[str],
                      **kwargs: Any) -> subprocess.CompletedProcess:
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
        arguments = []
        for index, option in enumerate(options[1:]):
            arguments.append(f'"{option}"' if '--' in options[index] else option)
        print(f'Attempting to run with the following arguments: {" ".join(arguments)}')

    env = kwargs.get('env')
    if env:
        return subprocess.run(options, capture_output=True, check=True, env=env)

    if kwargs.get('cwd'):
        return subprocess.run(options, capture_output=True, check=True, cwd=kwargs.get('cwd'))

    return subprocess.run(options, capture_output=True, check=True)


def handles_console_error(exit_code_on_error: int = 1,
                          ok_codes: tuple[int, ...] = (0,)) -> Callable:
    """
    Decorator to handle console tool errors.

    Args:
        exit_code_on_error (int): Exit code to use when an error occurs.
        ok_codes (tuple[int, ...]): Exit codes considered as success. Defaults to (0,).

    Returns:
        Callable: The wrapped function.
    """

    def decorator(func: Callable) -> Callable:
        """
        Decorator to handle console tool errors.

        Args:
            func (Callable): The function to be decorated, expected to return a `CompletedProcess`.

        Returns:
            Callable: The wrapped function with error handling.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> None:
            """
            Wrapper function to handle console tool errors.

            Args:
                *args (Any): Variable length argument list to pass to the decorated function.
                **kwargs (Any): Arbitrary keyword arguments to pass to the decorated function.
            """
            try:
                result = func(*args, **kwargs)
            except subprocess.CalledProcessError as error:
                print(f"Exit code: {error.returncode}.")
                log_output('Console run stdout', error.output)
                # return code is not 0, but sometimes it still can be OK
                if error.returncode in ok_codes:
                    print(f"Exit code: {error.returncode}.")
                    log_output('Console run stdout', error.output)
                else:
                    print(f"Check failed with exit code {error.returncode}.")
                    log_output('Console run stderr', error.stderr)
                    sys.exit(exit_code_on_error)
            else:
                print(f"Exit code: {result.returncode}.")
                log_output('Console run stdout', result.stdout)

        return wrapper

    return decorator
