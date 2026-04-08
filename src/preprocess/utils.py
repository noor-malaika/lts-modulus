"""
Logger and utility functions for data processing.
"""
import re
import logging
from pathlib import Path
from functools import reduce

def setup_logger(file_name="dataset"):
    logger = logging.getLogger("DatasetLogger")
    logger.setLevel(logging.DEBUG)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)

    # File Handler
    fh = logging.FileHandler(f"logs/{file_name}.log")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def fix_scientific_notation(value: str) -> float:
    """Convert incorrectly formatted scientific notation to proper float."""
    corrected_value = re.sub(r"(?<=\d)-(\d+)$", r"e-\1", value)
    return float(corrected_value)

def get_files_from_var_dirs(base_dir, variant, logger):
    var_files = []
    patterns = ["FYP*.nas", "*.pch", "CBUSH*.nas", "*.fem"]
    try:
        for var_dir in Path(base_dir).glob(variant):
            matching_files = [
                next(var_dir.rglob(pattern), None) for pattern in patterns
            ]
            var_files = [str(file) for file in matching_files if file is not None]
            assert len(var_files) == 4
        return var_files
    except Exception as e:
        logger.error(f"Can't get files {e}")

def get_nested_value(data, keys, default=None):
    try:
        return reduce(lambda d, key: d[key], keys, data)
    except (KeyError, TypeError):
        return default
