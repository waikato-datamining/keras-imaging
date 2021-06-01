def factorial(n: int, m: int = 1) -> int:
    """
    Returns the multiplication of all positive integers from [n] down
    to (but not including) [m].
    """
    if m < 1:
        raise ArithmeticError(f"m cannot be less than 1, got {m}")
    if n <= m:
        raise ArithmeticError(f"n must be greater than m, got m = {m}, n = {n}")

    result = n
    n -= 1
    while n > m:
        result *= n
        n -= 1

    return result


def n_choose_m(n: int, m: int, order_matters: bool = False):
    """
    Gets the number of ways to choose [m] items from a pool of
    [n] possibilities.
    """
    if m > n or m < 0:
        return 0
    if m == n or m == 0:
        return 1

    n_minus_m = n - m

    if order_matters:
        return factorial(n, n_minus_m)

    return (
        factorial(n, m) / factorial(n_minus_m)
        if m > n_minus_m else
        factorial(n, n_minus_m) / factorial(m)
    )
