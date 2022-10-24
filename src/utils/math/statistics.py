# the purpose of this file is to house mathematical functions that
# probably already exist in well known python libraries, but are reproduced
# here for learning purposes
import math


def choose(n: int, x: int) -> int:
    """Combinatorics 'choose' formula"""
    num_ordered_arrangements = math.factorial(n) / math.factorial(n - x)
    num_unordered_arrangements = num_ordered_arrangements // math.factorial(x)
    return int(num_unordered_arrangements)
