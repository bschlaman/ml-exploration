import csv
import logging
import random
import itertools
from typing import Generator
from colorama import Fore

from utils.helpers import data_print
from constants import RANDOM_SEED

log = logging.getLogger(__name__)


def get_data_stats():
    data = _fetch_data_from_file()
    labeled_data = {
        "num datapoints": len(data),
        "num unique urls": len({_["news_url"] for _ in data}),
        "num unique words": len(
            set(itertools.chain(*(_["title"].split() for _ in data)))
        ),
    }
    for line in data_print(labeled_data, Fore.YELLOW):
        log.info(line)


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
        return list(reader)
