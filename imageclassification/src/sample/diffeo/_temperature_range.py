import math


def temperature_range(n: int, c: int):
    """
    Determines the minimum and maximum temperatures that will result in a
    diffeomorphism given image size n and cut-off mode c. See Appendix B
    of the paper for details.

    Modified from function 'temperature_range' at
    https://github.com/pcsl-epfl/diffeomorphism/blob/0ca255c179e56c3e43db75ff01677f13b94bfe4c/diff.py#L98.

    :param n:   The side-length of the image, in pixels.
    :param c:   The cut-off mode.
    :return:    The minimum and maximum temperature values.
    """
    log_c = math.log(c)
    T_min = 1 / (math.pi * n ** 2 * log_c)
    T_max = 4 / (math.pi ** 3 * c ** 2 * log_c)
    return T_min, T_max
