import csv
import logging
import random
from typing import Generator

log = logging.getLogger(__name__)


def data_iter() -> Generator[dict[str, str], None, None]:
    data = _fetch_data_from_file()
    random.shuffle(data)
    yield from data


def _fetch_data_from_file() -> list[dict[str, str]]:
    with open("../data/data.csv") as csvfile:
        # for now, read the data into memory
        # public methods should appear as generators to their consumers
        reader = csv.DictReader(csvfile)
        log.debug(f"using data fields: {reader.fieldnames}")
        return list(reader)[:10000]
