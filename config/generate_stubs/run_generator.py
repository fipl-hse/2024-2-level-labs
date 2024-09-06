"""
Runner for generating and auto-formatting stubs.
"""

import sys
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.generate_stubs.generator import ArgumentParser, NoDocStringForAMethodError


def remove_implementation(source_code_path: Path, res_stub_path: Path) -> None:
    """
    Wrapper for implementation removal from a listing.

    Args:
        source_code_path (Path): Path to source code
        res_stub_path (Path): Path to resulting stub
    """
    stub_generator_path = Path(__file__).parent / 'generator.py'
    args = [
        str(stub_generator_path),
        '--source_code_path',
        str(source_code_path),
        '--target_code_path',
        str(res_stub_path)
    ]
    res_process = _run_console_tool(str(choose_python_exe()), args, debug=False)
    print(res_process.stdout.decode('utf-8'))
    if res_process.returncode != 0:
        raise NoDocStringForAMethodError(res_process.stderr.decode('utf-8'))


def format_stub_file(res_stub_path: Path) -> None:
    """
    Autoformat resulting stub.

    Args:
        res_stub_path (Path): Path to resulting path
    """
    args = [
        '-m',
        'black',
        '-l',
        '100',
        str(res_stub_path)
    ]
    res_process = _run_console_tool(str(choose_python_exe()), args, debug=False)
    if res_process.returncode != 0:
        raise ValueError(res_process.stderr.decode('utf-8'))


def sort_stub_imports(res_stub_path: Path) -> None:
    """
    Autoformat resulting stub.

    Args:
        res_stub_path (Path): Path to resulting stub
    """
    args = [
        str(res_stub_path)
    ]
    res_process = _run_console_tool('isort', args, debug=False)
    if res_process.returncode != 0:
        raise ValueError(res_process.stderr.decode('utf-8'))


def main() -> None:
    """
    Entrypoint for stub generation.
    """
    args = ArgumentParser().parse_args()

    source_code_path = Path(args.source_code_path).absolute()
    res_stub_path = Path(args.target_code_path).absolute()

    try:
        remove_implementation(source_code_path, res_stub_path)
    except NoDocStringForAMethodError as e:
        print(e)
        sys.exit(1)

    format_stub_file(res_stub_path)


if __name__ == '__main__':
    main()
