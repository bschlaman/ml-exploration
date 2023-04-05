import csv
import logging
import os

log = logging.getLogger(__name__)

RELATIVE_DATA_LAKE_PATH = "../../../../data"


def _construct_filepath(path_in_datalake: str) -> str:
    """Returns the relative file path that can be accessed
    from this utility"""
    rel_path = os.path.join(
        os.path.dirname(__file__), RELATIVE_DATA_LAKE_PATH, path_in_datalake
    )
    if not (os.path.isfile(rel_path) and os.access(rel_path, os.R_OK)):
        raise Exception(f"cannot read from file: {rel_path}")
    return rel_path


def load_from_file_csv(path_in_datalake: str):
    """Load data from a csv file into memory"""
    rel_path = _construct_filepath(path_in_datalake)
    with open(rel_path) as csvfile:
        log.debug(f"loaded data file: {path_in_datalake}")
        reader = csv.DictReader(csvfile)
        log.debug(f"loaded data fields: {reader.fieldnames}")
        return list(reader)


def load_from_file_bytes(path_in_datalake: str):
    """Load data file as bytes into memory"""
    rel_path = _construct_filepath(path_in_datalake)
    with open(rel_path, "rb") as f:
        log.debug(f"loaded data file: {path_in_datalake}")
        return f.read()
