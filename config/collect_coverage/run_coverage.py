"""
Runner for collecting coverage.
"""

import json
from pathlib import Path

from config.cli_unifier import _run_console_tool, choose_python_exe
from config.lab_settings import LabSettings


class CoverageRunError(Exception):
    """
    Error for coverage collection.
    """


class CoverageCreateReportError(Exception):
    """
    Error for report creation.
    """


def get_target_score(lab_path: Path) -> int:
    """
    Get student's expectations on a final mark.

    Args:
        lab_path (Path): Path to lab

    Returns:
        int: Desired score
    """
    settings = LabSettings(lab_path / 'settings.json')
    return settings.target_score


def extract_percentage_from_report(report_path: Path) -> int:
    """
    Load previous run value.

    Args:
        report_path (Path): Path to report

    Returns:
        int: Previous run value
    """
    with report_path.open(encoding='utf-8') as f:
        content = json.load(f)
    return int(content['totals']['percent_covered_display'])


def run_coverage_collection(lab_path: Path, artifacts_path: Path,
                            check_target_score: bool = True) -> tuple[int, bool]:
    """
    Entrypoint for a single lab coverage collection.

    Args:
        lab_path (Path): Path to lab
        artifacts_path (Path): Path to artifacts
        check_target_score (bool): Target score check

    Returns:
        tuple[int, bool]: Coverage percentage from report and True if we have failed tests,
         False in another case
    """
    print(f'Processing {lab_path} ...')

    python_exe_path = choose_python_exe()
    mark = ''
    if check_target_score:
        target_score = get_target_score(lab_path)
        mark = f' and mark{target_score}'

    args = [
        '-m', 'coverage', 'run',
        '--include', f'{lab_path.name}/main.py,'
                     f'{lab_path.name}/scrapper.py,'
                     f'{lab_path.name}/pipeline.py,'
                     f'{lab_path.name}/pos_pipeline.py',
        '-m', 'pytest', '-m', f'{lab_path.name}{mark}'
    ]
    res_process = _run_console_tool(str(python_exe_path), args,
                                    debug=True,
                                    cwd=str(lab_path.parent))

    stdout_decoded = res_process.stdout.decode('utf-8')
    print(res_process.stderr.decode('utf-8') + stdout_decoded)
    report_path = artifacts_path / f'{lab_path.name}.json'
    args = [
        '-m', 'coverage', 'json',
        '-o', str(report_path)
    ]
    res_process = _run_console_tool(str(python_exe_path), args,
                                    debug=True,
                                    cwd=str(lab_path.parent))
    print(res_process.stderr.decode('utf-8') + res_process.stdout.decode('utf-8'))
    return extract_percentage_from_report(report_path), 'failed' in stdout_decoded
