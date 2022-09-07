from utils.math import Vector2D
from utils.helpers import data_print
from colorama import Fore
import logging

log = logging.getLogger(__name__)


test_datapoints_class1 = [
    [0.11312343, 0.85504192],
    [0.21052602, 0.8820982],
    [0.18346975, 0.78740123],
    [0.10230091, 0.69811552],
    [0.22946542, 0.69540989],
    [0.33769052, 0.77116746],
    [0.31063425, 0.86857006],
    [0.25111044, 0.81986876],
    [0.29440048, 0.71976054],
    [0.15641347, 0.60341855],
    [0.188881, 0.6575311],
    [0.34851304, 0.58447915],
]
test_datapoints_class2 = [
    [0.78141347, 0.49519344],
    [0.83282039, 0.57095101],
    [0.80576412, 0.41673023],
    [0.70836152, 0.32473889],
    [0.5893139, 0.38696833],
    [0.68401087, 0.432964],
    [0.30522299, 0.25168694],
    [0.25652169, 0.34097266],
    [0.47567754, 0.19216313],
    [0.60554767, 0.19486876],
    [0.53790697, 0.29497699],
]
test_global_offset = 0.4


def init_dataset():
    # initialize the dataset with labels
    data = {}
    for x, y in test_datapoints_class1:
        data[(x - test_global_offset, y - test_global_offset)] = 1
    for x, y in test_datapoints_class1:
        data[(x - test_global_offset, y - test_global_offset)] = -1
    return data


def perceptron2D():
    # perceptron assumptions:
    # 1) Binary classification (i.e. yi∈{−1,+1})
    # 2) Data is linearly separable

    data = init_dataset()
    w = Vector2D(0, 0)
    while True:
        misses = 0
        for feature, label in data.items():
            vec = Vector2D(*feature)
            labeled_data = {
                "current w\u20D7": w,
                "miss count": misses,
                "testing feature": vec,
                "want": label,
                "got": w.dot_product(vec),
            }
            for line in data_print(labeled_data, Fore.BLUE):
                log.info(line)
            if label * w.dot_product(vec) <= 0:
                misses += 1
                log.info("miss! adjusting w\u20D7")
                vec.scale(label)
                w.add(vec)
            input("press a key to continue...")
        if not misses:
            break
