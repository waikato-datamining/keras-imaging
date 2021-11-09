from typing import Tuple

import numpy

from ._offset_norm import offset_norm


def noise(shape: Tuple[int, ...], norm: numpy.ndarray) -> numpy.ndarray:
    """
    Creates Gaussian noise of the given shape with the given norm.

    Modified from function 'diffeo_imgs' at
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L179.

    :param shape:   The shape of the noise array to create.
    :param norm:    The norm(s) that the created array should have.
    :return:        The noise array.
    """
    # Create noise with arbitrary norm
    unnormalised_noise = numpy.randn(shape)

    # Normalise it
    return unnormalised_noise / offset_norm(unnormalised_noise) * norm
