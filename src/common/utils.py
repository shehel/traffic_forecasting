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
