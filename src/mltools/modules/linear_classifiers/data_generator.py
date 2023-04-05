import collections
import logging
import random
from typing import Generator, Iterable

from mltools.utils.data import data_load

log = logging.getLogger(__name__)


def data_iter(num_records: int = 0) -> Generator[tuple[dict, bool], None, None]:
    """num_records reduces the data size for testing purposes"""
    data = data_load.load_from_file_csv("data.csv")
    random.shuffle(data)
    yield from transform_raw_data(data[-num_records:])


# TODO (2022.12.02): it may make more sense to include this in
# the Classifier classes, since the decision to randomly sample
# may not be universal across supervised learning algorithms
def _sample_features(features: list[str]) -> collections.Counter:
    """Select randomly from words and create feature vectors (counts)"""
    m = 12  # number of words per datum
    return collections.Counter(random.choices(features, k=m))


def transform_raw_data(raw_data: list[dict[str, str]]) -> Iterable[tuple[dict, bool]]:
    """Convert raw data into the format the linear classifiers expect.
    Format expected:
    ( {'word1': count1, 'word2': count2, ...}, class )
    """

    label_key = "real"
    features_key = "title"

    def _conv_xy(d):
        x, y = d
        words = list(
            # word after non alnum char extraction must not be empty
            filter(
                bool, map(lambda w: "".join(filter(str.isalnum, w)), x.lower().split())
            )
        )
        return (_sample_features(words), bool(int(y)))

    return map(_conv_xy, ((rd[features_key], rd[label_key]) for rd in raw_data))
