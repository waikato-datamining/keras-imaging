import numpy

from ._displacement_field import displacement_field
from ._Interpolation import Interpolation


def deform_image_with_diffeomorphism(
        images: numpy.ndarray,
        c: int,
        T: float,
        interpolation: Interpolation
) -> numpy.ndarray:
    """
    Deforms images by applying a random diffeomorphism to them.

    N.B. Applies the __SAME__ diffeomorphism to all images. For a different diffeomorphism
         per image, loop over the images, calling this function on each individually.

    Modified from functions 'deform' and 'remap' at
    https://github.com/pcsl-epfl/diffeomorphism/blob/0ca255c179e56c3e43db75ff01677f13b94bfe4c/diff.py#L34
    and
    https://github.com/pcsl-epfl/diffeomorphism/blob/0ca255c179e56c3e43db75ff01677f13b94bfe4c/diff.py#L59
    respectively.

    :param images:          The image(s) to deform, as an array of dimension [..., n, n].
    :param c:               The cut-off mode.
    :param T:               The displacement 'Temperature'.
    :param interpolation:   The type of interpolation to apply.
    :return:                The deformed images.
    """
    # Get the image side-length, ensuring it is square
    n = images.shape[-1]
    assert images.shape[-2] == n, 'Image(s) should be square.'

    # Create the displacement fields for the two image axes
    dx = displacement_field(n, c, T)
    dy = displacement_field(n, c, T)

    # Create image pixel indices
    indices = numpy.arange(n, dtype=dx.dtype)
    y, x = numpy.meshgrid(indices, indices)

    # Apply displacement
    xn = (x - dx).clamp(0, n-1)
    yn = (y - dy).clamp(0, n-1)

    # Remainder of code (applying interpolation) is verbatim from source (linked above)
    if interpolation == Interpolation.BILINEAR:
        xf = xn.floor().long()
        yf = yn.floor().long()
        xc = xn.ceil().long()
        yc = yn.ceil().long()

        xv = xn - xf
        yv = yn - yf

        return (1-yv) * (1-xv) * images[..., yf, xf] + (1 - yv) * xv * images[..., yf, xc] + yv * (1 - xv) * images[..., yc, xf] + yv * xv * images[..., yc, xc]

    elif interpolation == Interpolation.GAUSSIAN:
        # can be implemented more efficiently by adding a cutoff to the Gaussian
        sigma = 0.4715

        dx = (xn[:, :, None, None] - x)
        dy = (yn[:, :, None, None] - y)

        c = (-dx**2 - dy**2).div(2 * sigma**2).exp()
        c = c / c.sum([2, 3], keepdim=True)

        return (c * images[..., None, None, :, :]).sum([-1, -2])

    else:
        raise Exception(f"Unknown interpolation method: {interpolation}")