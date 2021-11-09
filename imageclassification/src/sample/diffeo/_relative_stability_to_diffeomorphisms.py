from typing import Callable, Tuple

import numpy

from ._deform_image_with_diffeomorphism import deform_image_with_diffeomorphism
from ._Interpolation import Interpolation
from ._offset_norm import offset_norm
from ._noise import noise


def relative_stability_to_diffeomorphisms(
    f: Callable[[numpy.ndarray], numpy.ndarray],
    images: numpy.ndarray,
    T: float,
    c: int,
    interpolation: Interpolation,
    normalise: bool = False
) -> Tuple[float, float]:
    """
    Calculates the relative stability to diffeomorphisms Rf for a function f, as in Equation 4.

    Modified from functions 'diffeo_imgs' and 'relative_distance' at
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L132
    and
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L203
    respectively.

    :param f:               The predictor function to test.
    :param images:          The test-set images to use to determine f's stability.
    :param T:               The diffeo temperature value to use.
    :param c:               The cut-off mode.
    :param interpolation:   The interpolation method to use.
    :param normalise:       Whether to normalise Df, Gf as in Equation 5.
    :return:                (Df, Gf), f's stability to diffeomorphisms and noise respectively. Rf = Df / Gf.
    """
    # Evaluate the function on the unmodified images
    unmodified_predictions = f(images).reshape(len(images), -1)

    # Create diffeomorphed versions of each image
    diffeo_deformed_images = numpy.array([deform_image_with_diffeomorphism(image, c, T, interpolation) for image in images])

    # Create noisy versions of each image with the same noise norm as the diffeo norm
    diffeo_norm = offset_norm(diffeo_deformed_images - images)
    noise_deformed_images = images + noise(images.shape, diffeo_norm)

    # Evaluate the function on both sets of distorted images
    diffeo_predictions = f(diffeo_deformed_images).reshape(len(images), -1)
    noise_predictions = f(noise_deformed_images).reshape(len(images), -1)

    # Calculate the unnormalised stabilities
    D_f = (diffeo_predictions - unmodified_predictions).pow(2).median(0).values.sum().item()
    G_f = (noise_predictions - unmodified_predictions).pow(2).median(0).values.sum().item()

    # Normalise them if selected
    if normalise:
        normalisation_factor = numpy.linalg.norm(
            unmodified_predictions[:, None, :] - unmodified_predictions[None, :, :],
            axis=2
        )
        if normalisation_factor == 0.0:
            normalisation_factor = 1e-10
        D_f = D_f / normalisation_factor
        G_f = G_f / normalisation_factor

    return D_f, G_f
