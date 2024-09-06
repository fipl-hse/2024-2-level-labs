"""
Check the relevance of stubs.
"""
# pylint: disable=too-many-locals, too-many-statements
import sys
from pathlib import Path

from config.constants import PROJECT_CONFIG_PATH
from config.generate_stubs.generator import cleanup_code
from config.generate_stubs.run_generator import format_stub_file, sort_stub_imports
from config.project_config import ProjectConfig


def get_code(code_path: Path) -> str:
    """
    Get clear code from file.

    Args:
        code_path (Path): Path to file with code

    Returns:
        str: Clear code
    """
    with code_path.open(encoding='utf-8') as file:
        code_text = file.read()
    return code_text


def clear_examples(lab_path: Path) -> None:
    """
    Clean temp files.

    Args:
        lab_path (Path): Path to temp files
    """
    for module_name in get_module_names():
        expected_stub_path = lab_path / f'example_{Path(module_name).stem}_stub.py'
        expected_stub_path.unlink(missing_ok=True)


def get_module_names() -> tuple[str, ...]:
    """
    Get names of potential candidates for stub generation.

    Returns:
        tuple[str, ...]: Names
    """
    return (
        'main.py',
        'start.py',
        'service.py',
        'scrapper.py',
        'pipeline.py',
    )


def main() -> None:
    """
    Check the relevance of stubs.
    """
    project_config = ProjectConfig(PROJECT_CONFIG_PATH)
    labs_paths = project_config.get_labs_paths()
    code_is_equal = True
    for lab_path in labs_paths:
        print(f'Processing {lab_path}...')
        if lab_path.name == 'core_utils':
            print(f'\t\tIgnoring {lab_path} as it can\'t have stubs.')
            continue

        for module_name in get_module_names():
            module_path = lab_path / module_name
            if not module_path.exists():
                print(f'\t\tIgnoring {module_path} as it does not exist.')
                continue
            print(f'\t{module_path}')

            stub_file_name = f'{module_path.stem}_stub.py'
            stub_code = get_code(module_path.parent / stub_file_name)
            actual_code = cleanup_code(module_path)

            expected_stub_content_path = module_path.parent / f'example_{stub_file_name}'

            with expected_stub_content_path.open(mode='w', encoding='utf-8') as file:
                file.write(actual_code)
            format_stub_file(expected_stub_content_path)
            sort_stub_imports(expected_stub_content_path)

            expected_formatted_stub = get_code(expected_stub_content_path)

            if stub_code != expected_formatted_stub:
                code_is_equal = False
                print(f'\tYou have different {module_path.name} and its stub in {lab_path}')

            if lab_path.name == 'lab_8_llm':
                lab_7_main = get_code(lab_path / 'main.py')
                lab_8_main = get_code(lab_path.parent / 'lab_7_llm' / 'main.py')

                if lab_7_main != lab_8_main:
                    code_is_equal = False
                    print('You have different main for Lab 7 and Lab 8!')

        clear_examples(lab_path)
    if code_is_equal:
        print('All stubs are relevant')
    sys.exit(not code_is_equal)


if __name__ == '__main__':
    main()
