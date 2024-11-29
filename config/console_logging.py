"""
Root logger configuration etc.
"""

from logging import getLogger, Logger

from logging518.config import fileConfig

from config.constants import PROJECT_ROOT

fileConfig(PROJECT_ROOT / "pyproject.toml")


def get_root_logger() -> Logger:
    """
    Get the root logger of the project.

    Returns:
        Logger: Instance of root logger
    """
    return getLogger(" ")


def get_child_logger(file_path: str) -> Logger:
    """
    Get the child logger from root logger by path.

    Args:
        file_path (str): File path

    Returns:
        Logger: Instance of child logger
    """
    root_logger = get_root_logger()
    suffix = file_path.split(str(PROJECT_ROOT))[1]
    return root_logger.getChild(suffix)
