from abc import ABC
import logging
import sys

import torch

from utils.data import data_load

log = logging.getLogger(__name__)


encode = lambda s: [ord(c) for c in s]
decode = lambda tokens: "".join(chr(token) for token in tokens)


class DataWrapper(ABC):
    pass


def get_random_batch(data: torch.Tensor, block_size: int, batch_size: int):
    offsets = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in offsets])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in offsets])
    return x, y


def run():
    # redundant to convert to string and then back to bytes when encoding, but whatever
    data_raw = str(data_load.load_from_file_bytes("tinyshakespeare.txt"))
    log.info(f"data len: {len(data_raw)}, size: {sys.getsizeof(data_raw)}")

    torch.manual_seed(9)
    t = torch.tensor(encode(data_raw), dtype=torch.long)
    log.info(f"tensor details: {t.shape=} {t.dtype=}")
    print(t)

    # variables and functions with prefix 'native__' are
    # used to baseline training and prediction performance
    # against native python datastructures and methods
    native__data = encode(data_raw)

    # step 1) split the data
    split_index = int(0.8 * len(t))
    training_data, test_data = t[:split_index], t[split_index:]
    native__training_data, native__test_data = (
        native__data[:split_index],
        native__data[split_index:],
    )

    # step 2: define hyperparameters
    # block size: number of tokens processed at once.
    # With N datapoints, we have n=(N-1) examples (input-output pairs)
    # input(x1 .. xi) -> target(xi+1); i ∈ {1, n}
    # In this case, block_size is also the number of examples we
    # can generate, as well as the max context length.
    # batch size: number of examples processed in parallel
    block_size = 5
    batch_size = 4

    for _ in range(4):
        print("====================")
        x, y = get_random_batch(training_data, block_size, batch_size)
        print(x)
        print(y)
