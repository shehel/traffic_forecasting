#!/usr/bin/env python3
import dotenv
from pathlib import Path
import os
import logging
from typing import Dict, List, Optional


def t4c_apply_basic_logging_config(loglevel: str = None):
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO") if loglevel is None else loglevel,
        format="[%(asctime)s][%(levelname)s][%(process)d][%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
    )

# Adapted from
# https://github.com/lucmos/nn-template/blob/main/src/common/utils.py
def get_env(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely read an environment variable.
    Raises errors if it is not defined or it is empty.
    :param env_name: the name of the environment variable
    :param default: the default (optional) value for the environment variable
    :return: the value of the environment variable
    """
    if env_name not in os.environ:
        if default is None:
            raise KeyError(f"{env_name} not defined and no default value is present!")
        return default

    env_value: str = os.environ[env_name]
    if not env_value:
        if default is None:
            raise ValueError(
                f"{env_name} has yet to be configured and no default value is present!"
            )
        return default

    return env_value


def load_envs(env_file: Optional[str] = None) -> None:
    """
    Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# load environment variables
load_envs()

# set the cwd to the project root
PROJECT_ROOT: Path = Path(get_env("PROJECT_ROOT"))
DATA_PATH: Path = Path(get_env("DATA_PATH"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"
assert (
    DATA_PATH.exists()
), "You must configure the DATA_PATH environment variable in a .env file!"

os.chdir(PROJECT_ROOT)
