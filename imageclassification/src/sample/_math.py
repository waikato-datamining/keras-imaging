from typing import List, Set, Union


def factorial(
        of: int,
        down_to: int = 0
) -> int:
    """
    Returns the multiplication of all positive integers from ``of`` down
    to (but not including) ``down_to``.

    :param of:
                The greatest positive integer to include in the product.
    :param down_to:
                The greatest positive integer, less than ``of``, to exclude
                from the product.
    :return:
                The factorial of ``of`` down to ``down_to``. If ``of`` equals
                ``down_to``, the result is ``of``.
    """
    if down_to < 0:
        raise ArithmeticError(f"'down_to' cannot be less than 0, got {down_to}")
    if of < down_to:
        raise ArithmeticError(f"'of' must be at least 'down_to', got 'of = {of}, 'down_to' = {down_to}")

    result = 1
    while of > down_to:
        result *= of
        of -= 1

    return result


def number_of_subsets(
        set_size: int,
        subset_size: int,
        order_matters: bool = False,
        can_reselect: bool = False
):
    """
    Gets the number of ways to choose ``subset_size`` items from a set of
    ``set_size`` possibilities.


    """
    if set_size < 0 or subset_size > set_size or subset_size < 0:
        return 0
    if subset_size == 0:
        return 1

    if order_matters:
        return (
            set_size ** subset_size
            if can_reselect else
            factorial(set_size, set_size - subset_size)
        )

    remainder_size = set_size - subset_size

    if can_reselect:
        # TODO
        raise ArithmeticError(
            "Can't calculate the number of subsets when order doesn't matter and "
            "reselection is allowed"
        )

    return (
        factorial(set_size, subset_size) // factorial(remainder_size)
        if subset_size > remainder_size else
        factorial(set_size, remainder_size) // factorial(subset_size)
    )


def subset_to_subset_number(
        set_size: int,
        subset: Union[List[int], Set[int]]
) -> int:
    """
    TODO
    """
    subset_size = len(subset)
    if set_size < 0:
        raise ArithmeticError(f"'set_size' must be non-negative, got {set_size}")
    if subset_size > set_size:
        raise ArithmeticError(
            f"Size of 'subset' must be at most 'set_size', "
            f"got 'set_size' = {set_size}, 'subset' (length {subset_size}) = {subset}"
        )

    if subset_size == 0:
        return 0

    if isinstance(subset, list):
        if len(subset) != len(set(subset)):
            raise ArithmeticError(f"Duplicate items in subset: {subset}")
        result = subset[0]
        subset = [x - 1 if x > result else x for x in subset[1:]]
        factor = set_size - 1
        while len(subset) > 0:
            result *= factor
            next = subset[0]
            result += next
            factor -= 1
            subset = [x - 1 if x > next else x for x in subset[1:]]

        return result
    else:
        sorted = list(subset)
        sorted.sort(reverse=True)
        return sum(number_of_subsets(n, subset_size - k) for k, n in enumerate(sorted))


def subset_number_to_subset(
        set_size: int,
        subset_size: int,
        subset_number: int,
        order_matters: bool = False
) -> Union[List[int], Set[int]]:
    """
    TODO
    """
    if order_matters:
        subset = []
        factor = set_size - subset_size + 1
        while len(subset) < subset_size:
            next = subset_number % factor
            subset_number //= factor
            subset = [next] + [x + 1 if x >= next else x for x in subset]
            if order_matters:
                factor += 1

        return subset
    else:
        subset = set()
        num_subsets = number_of_subsets(set_size - 1, subset_size)
        k = subset_size
        for n in reversed(range(set_size)):
            if subset_number >= num_subsets:
                subset_number -= num_subsets
                subset.add(n)
                if len(subset) == subset_size:
                    break
                num_subsets = num_subsets * k // n
                k -= 1
            else:
                num_subsets = num_subsets * (n - k) // n

        return subset
