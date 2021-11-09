from sample.diffeo._Equation_2 import Equation_2


def displacement_field(n: int, c: int, T: float):
    """
    Calculates a random diffeomorphism displacement field in one dimension.

    Modified from lines 47-53 of function 'deform' at
    https://github.com/pcsl-epfl/diffeomorphism/blob/0ca255c179e56c3e43db75ff01677f13b94bfe4c/diff.py#L47.

    :param n:       The side-length of the image, in pixels.
    :param c:       The cut-off mode.
    :param T:       The displacement 'Temperature'.
    """
    # Calculate a randomised scalar field (normalised by √T)
    tau_u_over_sqrt_T = Equation_2(n, c)

    # Remove normalisation
    tau_u = tau_u_over_sqrt_T * (T ** 0.5)

    # Convert from image-size space to pixel-size space
    return tau_u * n
