"""
Generator for API docs for Sphinx.
"""
from pathlib import Path

from config.cli_unifier import _run_console_tool, handles_console_error
from config.constants import PROJECT_CONFIG_PATH
from config.project_config import ProjectConfig


@handles_console_error()
def run_sphinx_apidoc(args: list[str]) -> tuple[str, str, int]:
    """
    Run sphinx-apidoc with the specified arguments.

    Args:
        args (list[str]): Arguments for sphinx-apidoc.

    Returns:
        tuple[str, str, int]: stdout, stderr, exit code
    """
    return _run_console_tool('sphinx-apidoc', args, debug=False)


def generate_api_docs(labs_paths: list[Path],
                      apidoc_templates_path: Path,
                      overwrite: bool = False) -> None:
    """
    Generate API docs for all laboratory works.

    Iterate over the specified lab* folders under the source_code_root and
    generate the API .rst document in the lab folder, i.e.,
    source_code_root/lab_name/lab_name.api.rst.

    Args:
        labs_paths (list[Path]): Paths to labs
        apidoc_templates_path (Path): Path to apidoc templates
        overwrite (bool): Overwrite
    """
    for lab_path in labs_paths:
        lab_api_doc_path = lab_path

        args = [
            '-o',
            str(lab_api_doc_path),
            '--no-toc',
            '--no-headings',
            '--suffix',
            'api.rst',
            '-t',
            str(apidoc_templates_path),
            str(lab_path)
        ]
        if overwrite:
            args.insert(-1, '-f')

        excluded_paths = (lab_path.joinpath('tests'),
                          lab_path.joinpath('assets'),
                          lab_path.joinpath('start.py'),
                          lab_path.joinpath('helpers.py'))
        args.extend(map(str, excluded_paths))

        run_sphinx_apidoc(args)


if __name__ == '__main__':
    project_config = ProjectConfig(config_path=PROJECT_CONFIG_PATH)

    templates_path = Path(__file__).parent.joinpath('templates').joinpath('apidoc')

    generate_api_docs(labs_paths=project_config.get_labs_paths(),
                      apidoc_templates_path=templates_path,
                      overwrite=True)
