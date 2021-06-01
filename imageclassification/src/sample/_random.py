from random import Random
from typing import Set

from ._math import n_choose_m


def choice(n: int, m: int, rand: Random = Random()) -> Set[int]:
    """
    TODO
    """
    num_choices = n_choose_m(n, m, True)

    choice_number = rand.randrange(num_choices)

    result = set()

    while m > 0:
        next_chosen_index = choice_number % n

        while next_chosen_index in result:
            next_chosen_index += 1

        result.add(next_chosen_index)

        choice_number //= n
        n -= 1
        m -= 1

    return result
