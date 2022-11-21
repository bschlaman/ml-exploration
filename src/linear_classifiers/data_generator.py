import csv
import logging

log = logging.getLogger(__name__)


def data_gen():
    with open("../data/data.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        yield from reader
