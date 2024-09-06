"""
Runner for collecting coverage.
"""

import sys
from pathlib import Path
from typing import Iterable, Mapping, Optional

from config.collect_coverage.run_coverage import (CoverageCreateReportError, CoverageRunError,
                                                  run_coverage_collection)
from config.constants import PROJECT_CONFIG_PATH, PROJECT_ROOT
from config.lab_settings import LabSettings
from config.project_config import ProjectConfig

CoverageResults = Mapping[str, tuple[Optional[int], bool]]


def collect_coverage(all_labs_names: Iterable[Path],
                     artifacts_path: Path) -> CoverageResults:
    """
    Entrypoint for coverage collection for every required folder.

    Args:
        all_labs_names (Iterable[Path]): Names of all labs
        artifacts_path (Path): Path to artifacts

    Returns:
        CoverageResults: Coverage results
    """
    all_labs_results = {}
    for lab_path in all_labs_names:
        percentage = None
        has_fallen_tests = False
        try:
            if lab_path.name == 'core_utils':
                check_target = False
            else:
                check_target = True
            percentage, has_fallen_tests = run_coverage_collection(lab_path=lab_path,
                                                                   artifacts_path=artifacts_path,
                                                                   check_target_score=check_target)
        except (CoverageRunError, CoverageCreateReportError) as e:
            print(e)
        finally:
            all_labs_results[lab_path.name] = (percentage, has_fallen_tests)
    return all_labs_results


def is_decrease_present(all_labs_results: CoverageResults,
                        previous_coverage_results: dict) -> tuple[bool, bool, dict]:
    """
    Analyze coverage report versus previous runs.

    Args:
        all_labs_results (CoverageResults): Coverage results
        previous_coverage_results (dict): Previous coverage results

    Returns:
        tuple[bool, bool, dict]: Is decrease present or not, and failed tests present or not
    """
    print('\n\n' + '------' * 3)
    print('REPORT')
    print('------' * 3)
    any_degradation = False
    any_fallen_tests = False
    labs_with_thresholds = {}
    for lab_name, current_lab_args in all_labs_results.items():
        if current_lab_args[1]:
            any_fallen_tests = True
        current_lab_percentage = current_lab_args[0]
        prev_lab_percentage = previous_coverage_results.get(lab_name, 0)
        if current_lab_percentage is None:
            current_lab_percentage = 0
        diff = current_lab_percentage - prev_lab_percentage

        print(f'{lab_name:<30}: {current_lab_percentage}% ({"+" if diff >= 0 else ""}{diff})')
        labs_with_thresholds[lab_name] = current_lab_percentage
        if diff < 0:
            any_degradation = True
    print('\n\n' + '------' * 3)
    print('END OF REPORT')
    print('------' * 3 + '\n\n')

    return any_degradation, any_fallen_tests, labs_with_thresholds


def main() -> None:
    """
    Entrypoint for coverage collection.
    """
    artifacts_path = PROJECT_ROOT / 'build' / 'coverage'
    artifacts_path.mkdir(parents=True, exist_ok=True)

    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    coverage_thresholds = project_config.get_thresholds()

    all_labs_names = project_config.get_labs_paths(include_addons=True)

    not_skipped = []
    for lab_path in all_labs_names:
        if 'core_utils' not in lab_path.name:
            settings = LabSettings(lab_path / 'settings.json')
            if settings.target_score == 0:
                print(f'Skip {lab_path} as target score is 0')
                continue
        not_skipped.append(lab_path)

    all_labs_results = collect_coverage(not_skipped, artifacts_path)

    any_degradation, any_fallen_tests, labs_with_thresholds = \
        is_decrease_present(all_labs_results, coverage_thresholds)

    if any_degradation:
        print('Some of labs have worse coverage. We cannot accept this. Write more tests!')
        print('You can copy-paste the following content to the ./config/project_config.json '
              'to update thresholds. \n\n')

        project_config.update_thresholds(labs_with_thresholds)

        print(project_config.get_json())
        sys.exit(1)

    if any_fallen_tests:
        print("Some tests failed! We can't accept that.\n"
              "Make sure the tests pass. I wish you good luck! \n\n")
        sys.exit(1)

    print('Nice coverage. Anyway, write more tests!', end='\n\n')


if __name__ == '__main__':
    main()
