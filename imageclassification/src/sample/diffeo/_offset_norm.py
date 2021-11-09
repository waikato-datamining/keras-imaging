import numpy


def offset_norm(images: numpy.ndarray) -> numpy.ndarray:
    """
    Calculates the normalisation value for each image.

    Modified from function 'diffeo_imgs' at
    https://github.com/leonardopetrini/diffeo-sota/blob/15941397685cdb1aa3ffb3ee718f5a6dde14bab3/results/utils.py#L178.

    :param images:          The image(s), as an array of dimension [..., n, n].
    :return:                The normalisation factor of each image.
    """
    return (images ** 2).sum([1, 2, 3], keepdim=True).sqrt()
