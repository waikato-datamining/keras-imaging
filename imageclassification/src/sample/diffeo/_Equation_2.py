import numpy

from ._Equation_2_components import Equation_2_components


def Equation_2(
        n: int,
        c: int
) -> numpy.ndarray:
    """
    Calculates the τ_u (or equivalently τ_v) values from Equation 2 for all u/v,
    normalised by √T.

    Modified from function 'scalar_field' at
    https://github.com/pcsl-epfl/diffeomorphism/blob/0ca255c179e56c3e43db75ff01677f13b94bfe4c/diff.py#L25.

    :param n:       The side-length of the image, in pixels.
    :param c:       The cut-off mode.
    :return:        τ_u (or equivalently τ_v) for all u,v, normalised by √T.
    """
    # Calculate the deterministic components of the equation
    std_dev_C_over_sqrt_T, sinusoids = Equation_2_components(n, c)

    # Pick actual values for the Gaussian variables C_ij
    C_over_sqrt_T = numpy.random.randn(c, c) * std_dev_C_over_sqrt_T

    return numpy.einsum('ij,xi,yj->yx', C_over_sqrt_T, sinusoids, sinusoids)
