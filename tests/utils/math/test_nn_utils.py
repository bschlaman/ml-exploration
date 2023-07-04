import unittest
import numpy as np
from mltools.utils.math import neural_networks as nn


class TestOneHotEncoding(unittest.TestCase):
    def setUp(self):
        self.test_lables = np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3])
        self.ohY = nn.one_hot(self.test_lables)

    def test_one_hot_encoding_shape(self):
        assert self.ohY.shape == (len(self.test_lables), 10)

    def test_one_hot_encoding_structure(self):
        assert np.all(
            (self.ohY.sum(axis=1) == 1)
            & np.all((self.ohY == 0) | (self.ohY == 1), axis=1)
        )

    def test_one_hot_encoding_values(self):
        assert self.ohY[0, 3] == 0
        assert self.ohY[0, 5] == 1
        assert self.ohY[4, 9] == 1

class TestCrossEntropyLoss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_a = np.array([
            [1,2,3],
            [4,-5,-6],
            [-0.3, 0.1, 0.009],
        ])
    
    def test_categorical_cross_entropy_loss(self):
        pass