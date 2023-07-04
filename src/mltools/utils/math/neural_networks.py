import numpy as np

def one_hot(Y: np.ndarray) -> np.ndarray:
    """Constructs a matrix whose rows represent
    the target probability distribution of categories
    for each row in Y, which in this case will be a
    1 in the column indexed by the desired label,
    and zeros everywhere else.
    """
    one_hot_Y = np.zeros((Y.size, np.max(Y) + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y

def categorical_cross_entropy_loss(ohY: np.ndarray, A: np.ndarray) -> float:
    assert np.all(
        (ohY.sum(axis=1) == 1) & np.all((ohY == 0) | (ohY == 1), axis=1)
    )  # one-hot encoding
    return -np.sum(ohY * np.log(A), dtype=float)

def softmax(A: np.ndarray):
    return np.exp(A) / np.sum(np.exp(A))
