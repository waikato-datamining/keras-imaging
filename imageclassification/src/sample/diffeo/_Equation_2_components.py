import functools
from typing import Tuple

import numpy


@functools.lru_cache()
def Equation_2_components(
        n: int,
        c: int,
        dtype=numpy.float64
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculates the C and sin(...) values from Equation 2 for all i/j/u/v.

    Modified from function 'scalar_field_modes' at
    https://github.com/pcsl-epfl/diffeomorphism/blob/0ca255c179e56c3e43db75ff01677f13b94bfe4c/diff.py#L12.

    :param n:  The side-length of the image, in pixels.
    :param c:  The cut-off mode.
    :param dtype:   The Numpy data-type to use for calculations.
    :return:
        - the standard deviation of the Gaussian variables C_ij, normalised by √T, for all i,j.
        - sin(iπu) for all i,u (or sin(jπv) for all j,v, which is identical).
    """
    # The u/v variables are image positions, normalised by the side-length of the image
    u_or_v = numpy.linspace(0, 1, n, dtype=dtype)

    # The sinusoidal modes considered, from the harmonic up to (and including) the cut-off given
    modes = numpy.arange(1, c + 1, dtype=dtype)

    # Create mode indices for performing sum
    i, j = numpy.meshgrid(modes, modes)

    # The following two calculations are performed slightly differently to the source, so we don't need
    # the 0.5 offset for numerical precision reasons. The original code looked like:
    # sqrt_i2_plus_j2 = (i.pow(2) + j.pow(2)).sqrt()
    # std_dev_C_over_sqrt_T = (sqrt_i2_plus_j2 < c + 0.5) / sqrt_i2_plus_j2

    # Calculate the value to compare to the cut-off constraint (also T / C^2).
    i2_plus_j2 = i.pow(2) + j.pow(2)

    # Calculate the magnitude of the mode, normalised by T
    std_dev_C_over_sqrt_T = ((i2_plus_j2 <= (c ** 2)) / i2_plus_j2).sqrt()

    # Calculate the values of the sinusoids for each mode at each pixel position
    sinusoids = numpy.sin(numpy.pi * u_or_v[:, None] * modes[None, :])

    return std_dev_C_over_sqrt_T, sinusoids
